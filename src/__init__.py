"""星辰科技企业制度问答系统 —— 核心业务代码包。

模块规划（随开发步骤逐步落地）：
    config.py      —— 统一配置：路径、LM Studio 地址、模型名、检索参数
    ingest.py      —— 读取 data/docs/ 下的 .md/.txt/.docx/.pdf 并切分为文本块（第二步）
    vectorstore.py —— 封装 ChromaDB 的写入与检索（第二步）
    rag_chain.py   —— 检索 + 调用本地模型生成带来源标注的回答（第三步）
"""
