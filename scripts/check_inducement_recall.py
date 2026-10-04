"""临时排查脚本：诱导性查询测试 —— 检验纯标题块是否会被误召回。

问题背景：库里有 6 个"纯标题块"（只有文档标题 + (2026版) 副标题，没有正文）。
如果它们会在检索中被召回，就会占掉 Top-K 名额、稀释给模型的参考资料质量。

测试方法：刻意用"问整份文档主题"的方式提问（这类问题与"文档标题"字面高度相似，
是最容易把标题块钓出来的问法），各跑一次 Top-5 检索，看这 6 个块有没有进榜。

判定目标块时复用 scripts/check_title_only.py 的同一套标准，避免两处口径不一致。

运行（需要 LM Studio 已启动并加载向量模型）：
    .venv/Scripts/python.exe scripts/check_inducement_recall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR.parent))

from check_title_only import SHORT_MAX, load_all_chunks  # noqa: E402

from src import config, vectorstore  # noqa: E402

TOP_K = 5

# 刻意选择"问整份文档讲了什么"的问法：与文档标题字面最接近，最容易钓出标题块
QUESTIONS = [
    "差旅与报销制度是什么？",
    "员工手册的主要内容是什么？",
    "财务发票管理制度讲了什么？",
    "信息安全管理制度有哪些要求？",
    "项目管理制度是怎么规定的？",
]


def detect_title_only_ids() -> dict[str, str]:
    """按与 check_title_only.py 相同的"口径二"找出纯标题块，返回 {chunk_id: 文本}。"""
    from check_title_only import RE_LIST, clean

    found: dict[str, str] = {}
    for text, meta in load_all_chunks():
        meta = meta or {}
        body = clean(text)
        if len(body) < SHORT_MAX and not RE_LIST.search(text):
            found[meta.get("chunk_id", "")] = body
    return found


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    targets = detect_title_only_ids()
    print("=" * 72)
    print(f"检测目标：库中的纯标题块（< {SHORT_MAX} 字且不含列表符号），共 {len(targets)} 个")
    print("=" * 72)
    for cid, body in targets.items():
        print(f"  · {cid:<28} {body[:40]}")
    if not targets:
        print("  （库中已无纯标题块 —— 入库阶段的前置块过滤正在生效）")

    print("\n" + "=" * 72)
    print(f"诱导性查询测试：{len(QUESTIONS)} 个问题，各取 Top-{TOP_K}")
    print("=" * 72)

    summary: list[tuple[str, int, float | None]] = []

    for question in QUESTIONS:
        hits = vectorstore.similarity_search(question, k=TOP_K)
        print(f"\n【提问】{question}")
        print(f"  {'排名':<4}{'距离':<9}{'纯标题块':<10}来源 ｜ 章节")
        hit_count = 0
        for rank, (doc, distance) in enumerate(hits, start=1):
            meta = doc.metadata
            cid = meta.get("chunk_id", "")
            is_target = cid in targets
            hit_count += int(is_target)
            loc = f"{meta.get('source', '?')} ｜ {meta.get('section', '?')}"
            if meta.get("page"):
                loc += f"（第 {meta['page']} 页）"
            print(
                f"  {rank:<4}{distance:<9.4f}"
                f"{'⚠️ 是' if is_target else '否':<9}{loc}"
            )
        best = hits[0][1] if hits else None
        summary.append((question, hit_count, best))
        print(f"  → Top-{TOP_K} 中纯标题块数量：{hit_count}")

    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    print(f"  {'提问':<24}{'命中标题块':<12}{'Top-1 距离'}")
    for question, hit_count, best in summary:
        dist = f"{best:.4f}" if best is not None else "—"
        print(f"  {question:<22}{hit_count:<12}{dist}")

    total_hits = sum(h for _, h, _ in summary)
    print(f"\n  全部 {len(QUESTIONS)} 个问题、共 {len(QUESTIONS) * TOP_K} 个 Top-{TOP_K} 名额中，")
    print(f"  纯标题块被召回的次数：{total_hits}")
    if not targets:
        print("\n  ✅ 结论：库中已无纯标题块（入库过滤生效），Top-K 名额全部由真实内容块占据。")
    elif total_hits == 0:
        print("\n  ✅ 结论：纯标题块未被误召回，当前观察成立。")
    else:
        print("\n  ⚠️ 结论：纯标题块进入了 Top-5，需要评估是否过滤。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
