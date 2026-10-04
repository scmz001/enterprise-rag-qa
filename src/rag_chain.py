"""检索与生成：把用户提问变成一段"有据可查"的回答。

三层防编造设计：
    第一层（检索前）：距离预过滤。最近片段都离题太远时，直接返回固定话术，
                     连模型都不调用，既快又不会编。
    第二层（提示词）：明确要求模型只能依据资料作答、必须标注资料编号、
                     资料不足时必须原样回复固定话术。
    第三层（生成后）：检查模型输出，若发现它在说"查不到"，统一替换成标准话术；
                     若回答里没有任何来源编号，标记为"未标注来源"以便排查。

命令行用法：
    .venv/Scripts/python.exe -m src.rag_chain "年假几天？"     # 单次提问
    .venv/Scripts/python.exe -m src.rag_chain --demo           # 跑验收标准的四条
    .venv/Scripts/python.exe -m src.rag_chain -i               # 交互式问答
    .venv/Scripts/python.exe -m src.rag_chain -r "年假几天？"    # 只看检索结果，不调用模型
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import config, vectorstore

# ============================================================
# 一、提示词
# ============================================================

SYSTEM_PROMPT = """你是星辰科技的企业制度问答助手，只依据公司官方文档回答员工提问。

必须遵守的规则：
1. 【唯一依据】只能使用【参考资料】中的内容作答。严禁使用你自己的任何知识，
   严禁推测、联想或编造制度内容。
2. 【标注来源】每一条结论后面都要用方括号标注依据的资料编号，例如：年假为 5 天 [1]。
   引用多条时写作 [1][3]。
3. 【查不到就说查不到】如果【参考资料】不足以回答该问题，必须且只能回复这一句话：
   根据公司现有制度，无法确认该信息，请联系HR或者IT服务台。
   不要解释原因，不要给任何额外建议，不要列出资料里无关的内容。
4. 【数字必须一致】涉及金额、天数、时限、比例时，必须与资料完全一致，不得改写或估算。
5. 用简洁的中文回答，分条列出，不要写开场白和结束语。"""

USER_TEMPLATE = """【参考资料】
{materials}

【员工问题】
{question}"""


def _format_materials(sources: list["Source"]) -> str:
    """把检索到的片段拼成带编号和来源的参考资料块。"""
    blocks = []
    for s in sources:
        blocks.append(f"[{s.index}] 来源：{s.source} ｜ 章节：{s.section}\n{s.content}")
    return "\n\n".join(blocks)


# ============================================================
# 二、数据结构
# ============================================================


@dataclass
class Source:
    """一条被引用的参考资料。"""

    index: int          # 编号，对应回答里的 [1][2]
    source: str         # 文件名
    section: str        # 章节
    page: int | None    # 页码（仅 PDF 有）
    distance: float     # 与提问的距离，越小越相关
    content: str        # 片段原文

    @property
    def label(self) -> str:
        """给前端显示用的来源标题，如「员工手册.md ｜ 二、年假制度」。"""
        if self.page:
            return f"{self.source} ｜ {self.section}（第 {self.page} 页）"
        return f"{self.source} ｜ {self.section}"

    @property
    def snippet(self) -> str:
        """正文摘要，用于在界面上展示引用依据。"""
        body = re.sub(r"^#+\s*", "", self.content.strip(), flags=re.MULTILINE)
        body = " ".join(body.split())
        return body[:120] + ("…" if len(body) > 120 else "")


@dataclass
class Answer:
    """一次问答的完整结果。"""

    question: str
    text: str                                  # 给用户看的回答
    sources: list[Source] = field(default_factory=list)
    refused: bool = False                      # 是否走了"无法确认"兜底
    cited: list[Source] = field(default_factory=list)  # 回答里真正引用到的来源
    best_distance: float | None = None
    raw_text: str = ""                         # 模型原始输出，便于排查
    error: str = ""                            # 非空表示"生成失败"，而非"查不到"

    @property
    def has_citation(self) -> bool:
        return bool(self.cited)

    @property
    def ok(self) -> bool:
        """是否给出了可信结果（成功作答，或明确判定查不到）。"""
        return not self.error


# ============================================================
# 三、模型
# ============================================================


def get_llm(streaming: bool = False) -> ChatOpenAI:
    """构造指向 LM Studio 的对话模型。"""
    return ChatOpenAI(
        model=config.require_chat_model(),
        base_url=config.LM_STUDIO_BASE_URL,
        api_key=config.LM_STUDIO_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        timeout=config.REQUEST_TIMEOUT,
        streaming=streaming,
    )


# ============================================================
# 四、检索
# ============================================================


def retrieve(question: str, k: int | None = None) -> list[Source]:
    """检索相关片段并按编号封装；过滤掉距离过远的噪声。"""
    hits: list[tuple[Document, float]] = vectorstore.similarity_search(
        question, k=k or config.TOP_K
    )
    sources: list[Source] = []
    for doc, distance in hits:
        if distance > config.SCORE_THRESHOLD:
            continue  # 离题太远，不送给模型，避免它硬凑答案
        meta = doc.metadata
        sources.append(
            Source(
                index=len(sources) + 1,
                source=meta.get("source", "未知文件"),
                section=meta.get("section", "未知章节"),
                page=meta.get("page"),
                distance=float(distance),
                content=doc.page_content,
            )
        )
    return sources


# ============================================================
# 五、生成
# ============================================================

_RE_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_RE_CITE = re.compile(r"\[(\d+)\]")


def _clean_model_output(text: str) -> str:
    """清掉推理模型的思考段、代码围栏和多余空白。"""
    text = _RE_THINK.sub("", text or "")
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    return text.strip()


def _looks_like_refusal(text: str) -> bool:
    """判断模型是不是在表达"资料里没有"。

    注意：空文本不算拒答。空输出是"生成失败"，必须走 GENERATION_ERROR_REPLY，
    绝不能伪装成"制度里没有这条规定"——那会误导员工。
    """
    compact = text.replace(" ", "")
    if not compact:
        return False
    patterns = ("无法确认该信息", "无法确认", "没有相关", "资料中没有", "联系HR", "请联系HR")
    return any(p in compact for p in patterns)


def _extract_citations(text: str, sources: list[Source]) -> list[Source]:
    """从回答里找出真正被引用的来源编号（按出现顺序去重）。"""
    by_index = {s.index: s for s in sources}
    cited: list[Source] = []
    for num in _RE_CITE.findall(text):
        src = by_index.get(int(num))
        if src and src not in cited:
            cited.append(src)
    return cited


def answer_question(question: str, k: int | None = None) -> Answer:
    """检索 + 生成，返回带来源标注的回答。"""
    question = (question or "").strip()
    if not question:
        raise ValueError("提问不能为空")

    sources = retrieve(question)
    best = min((s.distance for s in sources), default=None)

    # 第一层兜底：连一个够相关的片段都没有，直接返回固定话术，不调用模型
    if not sources:
        return Answer(
            question=question,
            text=config.NO_ANSWER_REPLY,
            sources=[],
            refused=True,
            best_distance=best,
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_TEMPLATE.format(
                materials=_format_materials(sources), question=question
            )
        ),
    ]
    response = get_llm().invoke(messages)
    raw = response.content
    finish_reason = (response.response_metadata or {}).get("finish_reason")
    text = _clean_model_output(raw)

    # 生成失败 ≠ 查不到：模型空输出通常是"思考耗尽了输出额度"或服务异常，
    # 必须单独报错，不能让员工以为制度里没有相关规定。
    if not text:
        if finish_reason == "length":
            reason = "模型把全部输出额度都用于内部推理，未能给出回答（可调大 config.LLM_MAX_TOKENS）"
        else:
            reason = f"模型返回了空内容（finish_reason={finish_reason}）"
        return Answer(
            question=question,
            text=config.GENERATION_ERROR_REPLY.format(reason=reason),
            sources=sources,
            error=reason,
            best_distance=best,
            raw_text=raw or "",
        )

    # 第三层兜底：模型说"查不到"，统一成标准话术，保证口径一致
    if _looks_like_refusal(text):
        return Answer(
            question=question,
            text=config.NO_ANSWER_REPLY,
            sources=sources,
            refused=True,
            best_distance=best,
            raw_text=raw,
        )

    return Answer(
        question=question,
        text=text,
        sources=sources,
        refused=False,
        cited=_extract_citations(text, sources),
        best_distance=best,
        raw_text=raw,
    )


# ============================================================
# 六、命令行入口
# ============================================================

DEMO_QUESTIONS = [
    "年假几天？",                      # 验收标准：应基于《员工手册.md》回答
    "出差住宿超标了怎么办？",            # 验收标准：应基于《差旅与报销制度.docx》回答
    "P0 故障多久必须响应？",             # 验收标准：来源应精确到章节
    "公司年会在哪里举办？",              # 验收标准：文档中不存在，应回答"无法确认"
]


def _print_answer(ans: Answer, show_sources: bool = True) -> None:
    print(f"\n【提问】{ans.question}")
    if ans.best_distance is not None:
        print(f"（最近片段距离 {ans.best_distance:.4f}）")
    print(f"【回答】{ans.text}")
    if ans.error:
        return
    if not show_sources or not ans.sources:
        return
    if ans.refused:
        print("【参考过的片段】以下片段与问题最接近，但不足以作答：")
    elif ans.cited:
        print("【引用来源】")
    else:
        print("⚠️ 【引用来源】模型未标注来源编号，请人工核对：")
    shown = ans.cited or ans.sources
    for s in shown:
        print(f"  {s.label}")
        print(f"    {s.snippet}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    parser = argparse.ArgumentParser(description="星辰科技制度问答系统 —— 检索与生成")
    parser.add_argument("question", nargs="?", help="要提问的问题")
    parser.add_argument("--demo", action="store_true", help="跑验收标准的四个问题")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互式问答")
    parser.add_argument("-r", "--retrieval-only", action="store_true", help="只看检索，不调用模型")
    parser.add_argument("-k", type=int, default=None, help="检索片段数")
    args = parser.parse_args(argv)

    if args.retrieval_only:
        question = args.question or input("请输入问题：").strip()
        print(f"\n【提问】{question}")
        for s in retrieve(question, args.k):
            print(f"  [{s.index}] 距离 {s.distance:.4f} ｜ {s.label}")
            print(f"      {s.snippet}")
        return 0

    if args.demo:
        print("=" * 62)
        print("验收标准自测：四个问题")
        print("=" * 62)
        for q in DEMO_QUESTIONS:
            _print_answer(answer_question(q, k=args.k))
        return 0

    if args.interactive:
        print("输入问题回车即可，输入 exit 退出。")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in ("exit", "quit", "q"):
                break
            if question:
                _print_answer(answer_question(question, k=args.k))
        return 0

    if not args.question:
        parser.print_help()
        return 1
    _print_answer(answer_question(args.question, k=args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
