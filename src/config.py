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

# 文档开头、第一个章节标题之前的"前置块"最少要有多少实质文字才保留。
# 低于此值说明该块只有"文档标题 + (2026版) 副标题"、没有任何正文，属于纯标题块：
# 实测它们会被"这份制度讲了什么？"这类提问召回到第 1 名，占掉 Top-K 名额，
# 还曾让模型把 [1] 当成来源引用（点开却没有内容），因此在入库阶段就丢弃。
# 只对"前置块"生效，不会误伤内容较短的正常条款。
MIN_PREAMBLE_CHARS = 60

# ---------- 检索参数 ----------
COLLECTION_NAME = "startech_policy"  # ChromaDB 集合名
TOP_K = 4                            # 每次提问取回的相关片段数
# 距离阈值（余弦距离，越小越相关）。实测：文档内问题 0.31~0.38，文档外提问 0.49~0.63。
# 取 0.55 属于"宽松预过滤"：只拦截明显无关的提问，绝不误伤文档内问题；
# 真正的"查不到"由提示词约束模型来判断，因为语义不匹配是距离量不出来的。
SCORE_THRESHOLD = 0.55

# ---------- 生成参数 ----------
LLM_TEMPERATURE = 0.0     # 制度问答要的是稳定复述，不是创造力
# 输出额度上限。qwen3.5-9b 是"推理模型"，会先输出一大段内部思考再给答案，
# 这个额度必须同时覆盖"思考 + 正式回答"，且随问题复杂度增长：
#   具体事实型问题（"年假几天？"）      实测思考约 1010 token、总输出约 1100 → 曾设 800 时爆掉
#   整份文档概述型问题（"X制度是什么？"）实测思考约 2446 token、总输出约 2600 → 设 2048 时爆掉
# 额度是"上限"不是"目标"，调大不会拖慢简单问题（模型答完即停），
# 故直接给足到 4096。若某天仍被截断，界面会明确报错并提示调大此值。
LLM_MAX_TOKENS = 4096
REQUEST_TIMEOUT = 300     # 单次请求超时秒数（含模型思考时间）

# ---------- 固定话术（业务边界，禁止编造）----------
NO_ANSWER_REPLY = "根据公司现有制度，无法确认该信息，请联系HR或者IT服务台。"

# 生成失败时的提示。必须与 NO_ANSWER_REPLY 区分开：
# "服务出错"绝不能被显示成"制度里没有这条规定"。
GENERATION_ERROR_REPLY = (
    "抱歉，本次未能生成回答：{reason}\n"
    "这属于模型服务问题，**不代表公司制度中没有相关规定**，请重试或检查 LM Studio。"
)


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
        "距离阈值": SCORE_THRESHOLD,
    }


def require_chat_model() -> str:
    """对话模型未配置时给出明确的修复指引，而不是让程序报出难懂的错。"""
    if not CHAT_MODEL:
        raise RuntimeError(
            f"未配置对话模型。请打开 {ENV_FILE}，把 CHAT_MODEL= 后面填成 "
            "LM Studio 中已加载的对话模型名（可用 curl http://localhost:1234/v1/models 查看）。"
        )
    return CHAT_MODEL
