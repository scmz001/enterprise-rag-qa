"""数据加载与切分：把 data/docs/ 下的四类文档转成可检索的文本块。

处理思路（尽可能保留文档原始结构）：
    1. 每种格式先转成"带 markdown 标题标记"的文本：
         .md   —— 本身就是 markdown，原样使用
         .txt  —— 把「一、」「（一）」「1.1」这类中文序号提升为 ## / ### 标题
         .docx —— 读取 Word 段落样式，Title → #，Heading1 → ##
         .pdf  —— 提取文字层，并按同样的中文序号规则提升标题
    2. 用 MarkdownHeaderTextSplitter 按 #/##/### 层级切分，
       标题文字保留在正文里，并把「所属章节」写入元数据（用于回答时标注来源）。
    3. 仍超长的块再用中文友好的分隔符递归切分。

直接运行（不连接模型，只做解析与切分自检）：
    .venv/Scripts/python.exe -m src.ingest --dry-run
正式入库（需要 LM Studio 已加载向量模型）：
    .venv/Scripts/python.exe -m src.ingest
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from . import config

# ============================================================
# 一、通用小工具
# ============================================================

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_CN_NUM = "一二三四五六七八九十百零"

# 中文序号标题 → H2（二级标题）
_RE_H2 = [
    re.compile(rf"^第[{_CN_NUM}\d]+章"),
    re.compile(rf"^第[{_CN_NUM}\d]+节"),
    re.compile(rf"^[{_CN_NUM}]{{1,3}}、"),
]
# 中文序号标题 → H3（三级标题）
_RE_H3 = [
    re.compile(rf"^[（(][{_CN_NUM}]+[）)]"),
    re.compile(r"^\d+(?:\.\d+)+\s*"),
]
# 只在这些"运算符/括号/顿号"结尾时才认定必须接下一行：
# 真正的排版断行几乎都停在这类字符上；而以汉字或句末标点结尾的行，
# 属于独立段落，强行拼接会把标题和正文粘在一起。
_INCOMPLETE_END = "/\\×÷+-=*<>≤≥％%（(［[【｛{、,，"

# 页码占位标记，进入切分流程后再还原成 page 元数据
_PAGE_MARK = "<!--page:{}-->"
_RE_PAGE_MARK = re.compile(r"<!--page:(\d+)-->")


def _fix_console_encoding() -> None:
    """Windows 终端默认不是 UTF-8，直接打印中文会乱码或报错。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _read_text_file(path: Path) -> str:
    """读取纯文本文件，自动尝试常见中文编码。"""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _promote_headings(text: str) -> str:
    """把纯文本里的中文序号行提升为 markdown 标题。

    只处理"独占一行、且以中文序号开头"的行，正文中的列举项（如「1. 五险一金」）
    不会被误判成标题，因为 `1.` 这种阿拉伯数字单级序号不在识别范围内。
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            if any(p.match(stripped) for p in _RE_H2):
                stripped = "## " + stripped
            elif any(p.match(stripped) for p in _RE_H3):
                stripped = "### " + stripped
        out.append(stripped)
    return "\n".join(out)


def _plain_text(text: str) -> str:
    """去掉 markdown 标题标记并压平空白，用于衡量一个块里到底有多少实质文字。"""
    return " ".join(re.sub(r"^#+\s*", "", text, flags=re.MULTILINE).split())


def _ensure_title(text: str) -> str:
    """确保文档有且仅有一个一级标题（用首个非空行充当）。"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            if not line.lstrip().startswith("#"):
                lines[i] = "# " + line.strip()
            return "\n".join(lines)
    return text


# ============================================================
# 二、各格式 → 带标题标记的文本
# ============================================================


def _docx_node_text(node: ET.Element) -> str:
    """取出一个段落/单元格的全部文字。"""
    return "".join(t.text or "" for t in node.iter(_W_NS + "t"))


def _docx_heading_level(node: ET.Element) -> int:
    """根据段落样式判断标题级别：Title→1，Heading1→2，Heading2→3。"""
    ppr = node.find(_W_NS + "pPr")
    style = ppr.find(_W_NS + "pStyle") if ppr is not None else None
    if style is None:
        return 0
    val = (style.get(_W_NS + "val") or "").strip()
    if val.lower() in ("title", "标题"):
        return 1
    m = re.match(r"^(?:heading|标题)\s*(\d+)$", val, flags=re.IGNORECASE)
    if m:
        # 文档里的 Heading1 属于"章"，对应 markdown 的 ##
        return int(m.group(1)) + 1
    return 0


def _docx_blocks(body: ET.Element):
    """按文档顺序遍历顶层块（段落 / 表格），遇到内容控件则下钻一层。"""
    for node in body:
        if node.tag == _W_NS + "sdt":
            content = node.find(_W_NS + "sdtContent")
            if content is not None:
                yield from _docx_blocks(content)
        else:
            yield node


def load_docx(path: Path) -> str:
    """.docx 是 zip 包，直接解析 word/document.xml（纯标准库，无需额外依赖）。"""
    with zipfile.ZipFile(path) as zf:
        xml_bytes = zf.read("word/document.xml")
    body = ET.fromstring(xml_bytes).find(_W_NS + "body")
    if body is None:
        return ""

    lines: list[str] = []
    for node in _docx_blocks(body):
        if node.tag == _W_NS + "p":
            text = _docx_node_text(node).strip()
            if not text:
                continue
            level = _docx_heading_level(node)
            lines.append(f"{'#' * level} {text}" if level else text)
        elif node.tag == _W_NS + "tbl":
            # 表格按 markdown 管道符展开，避免制度里的标准表格丢失
            for row in node.iter(_W_NS + "tr"):
                cells = [_docx_node_text(c).strip() for c in row.findall(_W_NS + "tc")]
                if any(cells):
                    lines.append("| " + " | ".join(cells) + " |")
    return "\n\n".join(lines)


def _is_duplicate_head(prev: str, line: str) -> bool:
    """判断两行是不是"同一句话的重复标题"（PDF 里主标题/副标题常被拆成两行）。

    例如「星辰科技客户服务与数据安全政策」与
    「星辰科技客户服务与数据安全政策(2026版)」共享长前缀，属于两个独立段落，
    不能拼成一句。
    """
    head = min(len(prev), len(line), 8)
    return head >= 6 and prev[:head] == line[:head]


def _unwrap_pdf_lines(text: str) -> str:
    """修复 PDF 提取时的硬换行：把被排版切断的一句话接回去。

    只有「上一行停在运算符、括号或顿号这类未完结字符上」时才拼接；以汉字或
    句末标点结尾的行一律视为独立段落，避免把标题、列表项与正文粘成一坨。
    """
    merged: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            merged.append("")
            continue
        prev = merged[-1] if merged else ""
        should_join = (
            bool(prev)
            and prev[-1] in _INCOMPLETE_END
            and not prev.startswith("#")
            and not _is_duplicate_head(prev, line)
        )
        if should_join:
            merged[-1] = prev + line
        else:
            merged.append(line)
    return "\n".join(merged)


def load_pdf(path: Path) -> tuple[str, list[int]]:
    """提取 PDF 每页文字，返回（带页码标记的全文, 页码列表）。"""
    from pypdf import PdfReader  # 延迟导入，--dry-run 之外也只在处理 pdf 时才加载

    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _unwrap_pdf_lines(page.extract_text() or "")
        if text.strip():
            parts.append(_PAGE_MARK.format(i) + "\n" + text)
    return "\n\n".join(parts), list(range(1, len(reader.pages) + 1))


def document_to_markdown(path: Path) -> tuple[str, bool]:
    """把任意支持的文档转成带 markdown 标题的文本，返回（文本, 是否含页码标记）。"""
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        text = _read_text_file(path)
        if suffix == ".txt":
            text = _promote_headings(text)
        return _ensure_title(text), False
    if suffix == ".docx":
        text = load_docx(path)
        if text and not text.lstrip().startswith("#"):
            text = _ensure_title(text)          # 没有 Title 样式时用首行兜底
        return text, False
    if suffix == ".pdf":
        text, _ = load_pdf(path)
        # 提升中文序号标题；页码标记行不含序号，会被原样保留
        return _promote_headings(text), True
    raise ValueError(f"不支持的格式：{path.name}")


# ============================================================
# 三、加载 + 切分
# ============================================================


def list_document_files(docs_dir: Path | None = None) -> list[Path]:
    """列出待处理的文档，按文件名排序保证每次入库顺序一致。"""
    docs_dir = docs_dir or config.DOCS_DIR
    if not docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在：{docs_dir}")
    files = [
        p
        for p in sorted(docs_dir.iterdir())
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"{docs_dir} 中没有找到 {config.SUPPORTED_SUFFIXES} 文档")
    return files


def heading_splitter() -> MarkdownHeaderTextSplitter:
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc_title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,  # 标题留在正文里，检索时"年假制度"这类关键词更容易命中
    )


def char_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", "、", " ", ""],
        keep_separator=True,
    )


def split_document(path: Path) -> list[Document]:
    """单个文件 → 文本块列表（每块带文件、章节、页码等来源信息）。"""
    text, _ = document_to_markdown(path)
    if not text.strip():
        return []

    chunks = heading_splitter().split_text(text)
    current_page: int | None = None
    docs: list[Document] = []

    for chunk in chunks:
        content = chunk.page_content

        # 还原页码：标记按顺序出现，沿用最近一次出现的页码
        marks = _RE_PAGE_MARK.findall(content)
        if marks:
            current_page = int(marks[-1])
        content = _RE_PAGE_MARK.sub("", content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if len(content) < 15:            # 丢弃只有标题、没有正文的空块
            continue

        # 前置块（第一个章节标题之前的内容）若只剩"文档标题 + 版本号"，没有任何信息量，
        # 直接丢弃。否则它会在检索时抢走 Top-K 名额，甚至被模型当成来源引用。
        # 判定依据是"位置"（不属于任何章节）+ "长度"，不针对具体文件，换一批文档同样适用。
        if (
            "section" not in chunk.metadata
            and len(_plain_text(content)) < config.MIN_PREAMBLE_CHARS
        ):
            continue

        meta = {k: v for k, v in chunk.metadata.items() if v}
        meta["source"] = path.name
        meta["doc_title"] = meta.get("doc_title") or path.stem
        # section 兜底为文档标题，保证每块都能给出"章节"级别的引用
        meta["section"] = meta.get("section") or meta["doc_title"]
        if current_page is not None:
            meta["page"] = current_page
        docs.append(Document(page_content=content, metadata=meta))

    # 超长块继续按中文标点递归切分，元数据自动继承
    docs = char_splitter().split_documents(docs)

    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = f"{path.stem}-{i:03d}"
    return docs


def load_and_split_all(docs_dir: Path | None = None) -> list[Document]:
    """读取 docs 目录下所有文档并切分，返回全部文本块。"""
    all_docs: list[Document] = []
    for path in list_document_files(docs_dir):
        all_docs.extend(split_document(path))
    return all_docs


# ============================================================
# 四、命令行入口
# ============================================================


def _print_dry_run_report(docs: list[Document], files: list[Path]) -> None:
    by_source = Counter(d.metadata["source"] for d in docs)
    print("\n【1】文档读取情况")
    print(f"    共发现 {len(files)} 个文件，成功切出 {len(docs)} 个文本块")
    for path in files:
        print(f"    - {path.name:<32} {by_source.get(path.name, 0):>3} 块")

    print("\n【2】切分结果抽查（每份文档 1 条）")
    seen: set[str] = set()
    for doc in docs:
        src = doc.metadata["source"]
        if src in seen:
            continue
        seen.add(src)
        loc = doc.metadata.get("section", "")
        if doc.metadata.get("page"):
            loc += f"（第 {doc.metadata['page']} 页）"
        body = doc.page_content.replace("\n", " ")
        print(f"\n    ── {src} ｜ {loc}")
        print(f"       {body[:120]}{'…' if len(body) > 120 else ''}")

    lengths = [len(d.page_content) for d in docs]
    print("\n【3】文本块长度分布")
    print(
        f"    最短 {min(lengths)} 字 / 平均 {sum(lengths) // len(lengths)} 字 / "
        f"最长 {max(lengths)} 字（上限 {config.CHUNK_SIZE} 字）"
    )
    missing = [d.metadata["chunk_id"] for d in docs if not d.metadata.get("section")]
    print(f"    缺少章节信息的块：{len(missing)} 个")


def main(argv: list[str] | None = None) -> int:
    _fix_console_encoding()
    parser = argparse.ArgumentParser(description="星辰科技制度问答系统 —— 数据加载与入库")
    parser.add_argument("--dry-run", action="store_true", help="只做解析与切分，不连接 LM Studio")
    parser.add_argument("--stats", action="store_true", help="只查看向量库里现有数据")
    parser.add_argument("--keep", action="store_true", help="保留已有数据，追加写入（默认是全量重建）")
    args = parser.parse_args(argv)

    from . import vectorstore  # 延迟导入：--dry-run 时完全不碰向量库

    if args.stats:
        vectorstore.print_stats()
        return 0

    print("=" * 62)
    print("星辰科技制度问答系统 —— 第二步：数据加载与入库")
    print("=" * 62)
    for key, value in config.describe().items():
        print(f"  {key}：{value}")

    files = list_document_files()
    docs = load_and_split_all()

    if args.dry_run:
        _print_dry_run_report(docs, files)
        print("\n✅ 自检完成：文档解析与切分正常（未写入向量库）")
        return 0

    if not config.CHROMA_DIR.exists():
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.keep:
        vectorstore.reset_collection()
        print("\n已清空旧的向量数据，开始全量重建…")
    else:
        print("\n保留已有向量数据，开始追加…")

    written = vectorstore.index_documents(docs)
    print(f"\n✅ 入库完成：共写入并持久化 {written} 个文本块")
    vectorstore.print_stats()
    print("\n提示：以后只需重新运行本命令即可重建索引；查看现有数据用 --stats。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
