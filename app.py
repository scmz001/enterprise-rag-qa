"""星辰科技企业制度问答系统 —— Streamlit 前端。

启动：
    .venv/Scripts/streamlit run app.py

界面结构：
    侧边栏 —— 已加载文档清单、上传新文档、重建索引、运行信息
    主区域 —— 对话界面与历史记录，每条回答下方可展开查看引用来源
"""

from __future__ import annotations

import time

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from src import config, ingest, rag_chain, vectorstore

# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="星辰科技制度问答",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "年假几天？",
    "出差住宿超标了怎么办？",
    "发票抬头怎么写？",
    "P0 故障多久必须响应？",
]


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


# ============================================================
# 侧边栏
# ============================================================


def _render_document_list() -> None:
    """显示 data/docs/ 下的文档清单，以及每份文档已入库的文本块数量。"""
    st.subheader("📚 已加载文档")
    try:
        files = ingest.list_document_files()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    stats = vectorstore.collection_stats()
    indexed: dict[str, int] = stats.get("sources", {}) if stats.get("collection") else {}

    if stats.get("collection"):
        st.caption(f"知识库共 **{stats['count']}** 个文本块")
    else:
        st.warning("知识库为空，请点击下方「重建索引」")

    for path in files:
        n = indexed.get(path.name, 0)
        if n:
            st.markdown(f"✅ {path.name}  \n<small>　{n} 个文本块</small>", unsafe_allow_html=True)
        else:
            st.markdown(f"⚠️ {path.name}  \n<small>　尚未入库</small>", unsafe_allow_html=True)


def _handle_upload(uploaded) -> None:
    """把上传的文件保存进 data/docs/，然后重建索引。"""
    saved = []
    for file in uploaded:
        target = config.DOCS_DIR / file.name
        target.write_bytes(file.getbuffer())
        saved.append(file.name)
    st.success("已保存：" + "、".join(saved))
    _rebuild_index()


def _rebuild_index() -> None:
    """全量重建向量库：解析全部文档 → 重新向量化写入。"""
    try:
        with st.spinner("正在解析文档…"):
            docs = ingest.load_and_split_all()
        vectorstore.reset_collection()
        with st.spinner(f"正在向量化并写入 {len(docs)} 个文本块，请稍候…"):
            n = vectorstore.index_documents(docs)
        st.success(f"索引重建完成，共 {n} 个文本块。")
        st.rerun()
    except Exception as exc:  # 常见原因：LM Studio 没启动、向量模型没加载
        st.error(f"重建索引失败：{exc}")
        st.caption("请确认 LM Studio 已启动、且「Local Server」已开启并加载了向量模型。")


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("📘 制度问答")

        _render_document_list()

        st.divider()
        st.subheader("➕ 上传新文档")
        uploaded = st.file_uploader(
            "支持 .md / .txt / .docx / .pdf",
            type=["md", "txt", "docx", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button("保存并重建索引", use_container_width=True):
            _handle_upload(uploaded)

        if st.button("🔄 重建索引", use_container_width=True):
            _rebuild_index()

        st.divider()
        st.subheader("ℹ️ 运行信息")
        for key, value in config.describe().items():
            st.caption(f"{key}：{value}")

        st.divider()
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# 回答生成（含流式输出）
# ============================================================


def _build_messages(question: str, sources: list[rag_chain.Source]) -> list:
    return [
        SystemMessage(content=rag_chain.SYSTEM_PROMPT),
        HumanMessage(
            content=rag_chain.USER_TEMPLATE.format(
                materials=rag_chain._format_materials(sources), question=question
            )
        ),
    ]


def _stream_answer(messages: list) -> tuple[str, str]:
    """流式生成回答，返回（回答文本, 错误信息）。

    推理模型的特殊性：qwen3.5-9b 会先内部思考约 10~12 秒，这段时间接口只返回
    空内容，一个字都没有。因此不能只依赖"有字才更新界面"，否则界面会像卡死。
    这里用一个持续跳秒的状态条表示"还活着"，等第一段文字到达后切换为流式打字。
    """
    status = st.status("正在检索并推理…", expanded=False)
    holder = st.empty()

    buffer = ""
    started = time.time()
    last_tick = 0.0

    try:
        for chunk in rag_chain.get_llm(streaming=True).stream(messages):
            piece = chunk.content or ""
            if not piece:
                # 推理阶段：没有可见文字，用跳秒提示进度
                now = time.time()
                if now - last_tick > 0.4:
                    last_tick = now
                    status.update(
                        label=f"正在推理…已用时 {now - started:.0f} 秒"
                        "（qwen3.5-9b 会先内部思考，属正常现象）"
                    )
                continue
            if not buffer:
                status.update(label=f"推理完成（用时 {time.time() - started:.0f} 秒），正在输出…")
            buffer += piece
            holder.markdown(buffer + "▌")
    except Exception as exc:  # 服务未启动、模型未加载、超时等
        status.update(label="生成失败", state="error")
        return "", f"{type(exc).__name__}: {exc}"

    status.update(
        label=f"回答完成（总用时 {time.time() - started:.0f} 秒）",
        state="complete",
        expanded=False,
    )
    holder.markdown(buffer)
    return buffer, ""


def _render_sources(sources: list[rag_chain.Source], refused: bool) -> None:
    """每条回答下方的「引用来源」。"""
    if not sources:
        return
    title = "参考过的片段（不足以作答）" if refused else "引用来源"
    with st.expander(f"📎 {title}（{len(sources)} 条）", expanded=not refused):
        for src in sources:
            st.markdown(f"**{src.index}. {src.label}**")
            st.caption(f"相关度距离 {src.distance:.3f}（越小越相关）")
            st.markdown(f"> {src.snippet}")


def _answer(question: str) -> dict:
    """执行一次问答，返回可存入历史记录的消息字典。"""
    sources = rag_chain.retrieve(question)

    # 第一层兜底：没有足够相关的资料，直接给出固定话术，不调用模型
    if not sources:
        st.markdown(config.NO_ANSWER_REPLY)
        return {"role": "assistant", "content": config.NO_ANSWER_REPLY,
                "sources": [], "refused": True}

    text, error = _stream_answer(_build_messages(question, sources))

    if error:
        st.error(f"生成回答失败：{error}")
        st.caption("请确认 LM Studio 已启动并加载了对话模型 qwen3.5-9b。")
        return {"role": "assistant", "content": "", "sources": sources,
                "refused": False, "error": error}

    text = rag_chain._clean_model_output(text)

    if not text:
        st.warning(config.GENERATION_ERROR_REPLY.format(
            reason="模型没有返回任何内容（输出额度可能被内部推理耗尽）"))
        st.caption("这属于模型服务问题，不代表公司制度中没有相关规定。")
        return {"role": "assistant", "content": "", "sources": sources,
                "refused": False, "error": "empty"}

    # 第三层兜底：模型说"查不到"，统一成标准话术
    if rag_chain._looks_like_refusal(text):
        text = config.NO_ANSWER_REPLY
        st.markdown(text)
        _render_sources(sources, refused=True)
        return {"role": "assistant", "content": text, "sources": sources,
                "refused": True, "error": ""}

    cited = rag_chain._extract_citations(text, sources)
    if not cited:
        st.warning("模型未标注来源编号，以下内容请人工核对。")
    _render_sources(cited or sources, refused=False)
    return {"role": "assistant", "content": text, "sources": cited or sources,
            "refused": False, "error": ""}


# ============================================================
# 主区域
# ============================================================


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📘"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"], refused=msg.get("refused", False))


def _render_empty_hint() -> None:
    st.markdown(
        "我可以回答公司制度方面的问题，例如:\n"
        + "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS)
    )
    st.caption("回答只依据 data/docs/ 中的公司文档并标注来源；文档中没有的内容会明确告知无法确认。")


def main() -> None:
    _init_state()
    _render_sidebar()

    st.title("📘 星辰科技制度问答")
    st.caption("基于公司现有制度文档回答，并标注引用来源。")

    if not st.session_state.messages:
        _render_empty_hint()
    else:
        _render_history()

    question = st.chat_input("请输入你的问题，例如：年假几天？")
    if question:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        try:
            with st.chat_message("assistant", avatar="📘"):
                answer = _answer(question)
        except Exception as exc:
            st.error(f"处理失败：{exc}")
            st.caption("请确认 LM Studio 已启动、且 .env 中的模型名与 LM Studio 中加载的一致。")
            answer = {"role": "assistant", "content": "", "sources": [],
                      "refused": False, "error": str(exc)}

        st.session_state.messages.append(answer)


main()
