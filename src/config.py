"""统一配置：路径、LM Studio 连接信息、切分与检索参数。

所有可调项集中在此文件，其他模块只从这里取值，避免参数散落各处。
密钥与模型名从项目根目录的 .env 读取（.env 不纳入版本控制）。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根目录
DOCS_DIR = BASE_DIR / "data" / "docs"               # 原始文档目录（唯一知识来源）
STORAGE_DIR = BASE_DIR / "storage"                  # 本地数据目录
CHROMA_DIR = STORAGE_DIR / "chroma"                 # ChromaDB 持久化目录
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=False)

# ---------- LM Studio（OpenAI 兼容接口）----------
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").strip()
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio").strip()
CHAT_MODEL = os.getenv("CHAT_MODEL", "").strip()            # 对话模型，第三步使用
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3").strip()  # 向量模型

# ---------- 支持的文档格式 ----------
SUPPORTED_SUFFIXES = (".md", ".txt", ".docx", ".pdf")

# ---------- 切分参数 ----------
# 每个文本块的字符数上限。中文制度文档以短条款为主（多为 150~300 字），
# 500 字可让"一条制度"基本不被切断，同时保证向量语义聚焦。
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80  # 相邻块的重叠字数，防止关键句正好卡在切口处被割裂

# ---------- 检索参数 ----------
COLLECTION_NAME = "startech_policy"  # ChromaDB 集合名
TOP_K = 4                            # 每次提问取回的相关片段数

# ---------- 固定话术（业务边界，禁止编造）----------
NO_ANSWER_REPLY = "根据公司现有制度，无法确认该信息，请联系HR或者IT服务台。"


def describe() -> dict[str, object]:
    """返回当前配置摘要，供终端打印（不含任何密钥明文）。"""
    return {
        "文档目录": str(DOCS_DIR),
        "向量库目录": str(CHROMA_DIR),
        "LM Studio 地址": LM_STUDIO_BASE_URL,
        "对话模型": CHAT_MODEL or "（未配置，请填写 .env 的 CHAT_MODEL）",
        "向量模型": EMBEDDING_MODEL or "（未配置，请填写 .env 的 EMBEDDING_MODEL）",
        "配置文件": f"{ENV_FILE}（{'已找到' if ENV_FILE.exists() else '不存在，将使用默认值'}）",
        "切分大小/重叠": f"{CHUNK_SIZE} / {CHUNK_OVERLAP} 字",
        "检索条数": TOP_K,
    }
