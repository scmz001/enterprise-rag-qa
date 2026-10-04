"""临时排查脚本：确认向量库里哪些块是"纯标题块"（无实质正文）。

用途：验证 README「已知问题」中记录的"PDF 和 DOCX 的第一块为纯标题块（约 7 个）"。

直接读取 ChromaDB，不经过 embedding 接口，所以**不需要启动 LM Studio**。

运行：
    .venv/Scripts/python.exe scripts/check_title_only.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 允许用 `python scripts/xxx.py` 直接运行（此时 sys.path[0] 是 scripts/ 而非项目根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

SHORT_MAX = 60  # "短块"的字符数阈值

# 判定用的正则
RE_HEADING = re.compile(r"^#+\s*", re.MULTILINE)          # markdown 标题标记
RE_LIST = re.compile(                                      # 列表符号 / 编号开头
    r"(?:^|\n)\s*(?:[-*•·]|\d+[.、)]|[一二三四五六七八九十]+、)"
)
RE_DIGIT = re.compile(r"\d")                               # 阿拉伯数字
RE_SENT_END = re.compile(r"[。！？；;]$")                    # 以句末标点收尾


def clean(text: str) -> str:
    """去掉 markdown 标题标记并把空白压平，用于长度与内容判断。"""
    return " ".join(RE_HEADING.sub("", text).split())


def load_all_chunks() -> list[tuple[str, dict]]:
    """读取向量库中的全部文本块。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection(config.COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    return list(zip(data["documents"], data["metadatas"]))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    chunks = load_all_chunks()
    total = len(chunks)
    print(f"向量库共 {total} 个文本块（集合：{config.COLLECTION_NAME}）")

    # 先按长度筛出"短块"，再看它们各自踩中了哪几条排除条件
    rows = []
    for text, meta in chunks:
        body = clean(text)
        if len(body) >= SHORT_MAX:
            continue
        rows.append(
            {
                "source": (meta or {}).get("source", "未知"),
                "section": (meta or {}).get("section", ""),
                "page": (meta or {}).get("page"),
                "chunk_id": (meta or {}).get("chunk_id", ""),
                "raw": text,
                "body": body,
                "has_list": bool(RE_LIST.search(text)),
                "has_digit": bool(RE_DIGIT.search(body)),
                "has_end": bool(RE_SENT_END.search(body)),
            }
        )
    rows.sort(key=lambda r: len(r["body"]))

    # ---------- 口径一：严格按"无列表符号、无数字、不以句号结尾" ----------
    strict = [r for r in rows if not (r["has_list"] or r["has_digit"] or r["has_end"])]
    print("\n" + "=" * 70)
    print(f"【口径一】长度 < {SHORT_MAX} 字，且不含列表符号 / 数字 / 句末标点")
    print("=" * 70)
    if strict:
        for i, r in enumerate(strict, 1):
            loc = r["section"] + (f"（第 {r['page']} 页）" if r["page"] else "")
            print(f"\n{i}. 来源：{r['source']} ｜ 章节：{loc} ｜ 长度 {len(r['body'])} 字")
            print(f"   全文：{r['body']}")
    else:
        print("（无匹配）")

    # ---------- 口径二：只要求"长度 < 60 字且不是正常正文" ----------
    print("\n" + "=" * 70)
    print(f"【口径二】长度 < {SHORT_MAX} 字，排除条件放宽（仅排除含列表符号的块）")
    print("说明：带 (2026版) 之类数字的标题块会被口径一误排除，需看这份")
    print("=" * 70)
    loose = [r for r in rows if not r["has_list"]]
    for i, r in enumerate(loose, 1):
        loc = r["section"] + (f"（第 {r['page']} 页）" if r["page"] else "")
        marks = []
        if r["has_digit"]:
            marks.append("含数字")
        if r["has_end"]:
            marks.append("以句末标点结尾")
        note = f"  ← 口径一排除原因：{'、'.join(marks)}" if marks else "  ← 口径一也命中"
        first = "是" if r["chunk_id"].endswith("-000") else "否"
        print(f"\n{i}. 来源：{r['source']} ｜ 章节：{loc} ｜ 长度 {len(r['body'])} 字{note}")
        print(f"   编号：{r['chunk_id']} ｜ 是否该文件第一块：{first}")
        print(f"   全文：{r['body']}")

    # ---------- 横向对比：最短的 12 个块 ----------
    print("\n" + "=" * 70)
    print("【参考】库中最短的 12 个块（便于判断阈值是否合理）")
    print("=" * 70)
    print(f"{'长度':>5}  {'列表':<5}{'数字':<5}{'句末':<5} 来源")
    for r in rows[:12]:
        print(
            f"{len(r['body']):>5}  "
            f"{'是' if r['has_list'] else '否':<4} "
            f"{'是' if r['has_digit'] else '否':<4} "
            f"{'是' if r['has_end'] else '否':<4} "
            f"{r['source']}"
        )
    print(f"\n短块（< {SHORT_MAX} 字）合计：{len(rows)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
