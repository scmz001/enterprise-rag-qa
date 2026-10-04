# 星辰科技企业制度问答系统

一个基于 **RAG（检索增强生成）** 的企业制度问答应用：用自然语言提问，系统只依据 `data/docs/`
目录下的星辰科技官方文档作答，并在每条回答下标注引用来源；文档里查不到的内容，系统会明确回答
"无法确认"，绝不编造制度。

- 前端：Streamlit（单页应用，无需前后端分离）
- 编排：LangChain
- 向量库：ChromaDB（本地文件持久化，不需要单独启动服务）
- 大模型：LM Studio 本地部署模型（通过 OpenAI 兼容接口调用，数据不出本机）

---

## 一、目录结构

```
StarTech_RAG/
├── app.py                  # 【第四步】Streamlit 前端入口，一条命令启动整个应用
├── src/                    # 核心业务代码包
│   ├── __init__.py         # 包标识 + 模块规划说明
│   ├── config.py           # ✔ 统一配置：路径、LM Studio 地址、模型名、切分与检索参数
│   ├── ingest.py           # ✔ 读取 data/docs/ 四类文档并按标题层级切分成文本块
│   ├── vectorstore.py      # ✔ 封装 ChromaDB：向量入库 / 相似度检索 / 现状统计
│   └── rag_chain.py        # 【第三步】检索 + 生成，强制模型基于原文作答并标注来源
├── data/
│   └── docs/               # 企业原始文档（.md / .txt / .docx / .pdf），系统的唯一知识来源
├── storage/
│   └── chroma/             # ChromaDB 持久化目录，由第二步自动生成（内容不入 git）
├── requirements.txt        # 运行本项目所需的全部依赖库清单
├── .env.example            # 环境变量模板（复制为 .env 后填写 LM Studio 模型名）
├── .env                    # 本机真实配置与密钥，已被 .gitignore 忽略，不会提交
├── .gitignore              # 指定不纳入版本控制的文件（虚拟环境、密钥、缓存、向量数据）
├── prompts.md              # 项目需求与分步开发计划（开发依据）
└── README.md               # 本文件
```

---

## 二、依赖库说明（每个库的作用）

`requirements.txt` 中只包含运行本项目的必要库，逐条说明如下：

| 库 | 作用（一句话） |
| --- | --- |
| `streamlit` | 把 Python 脚本直接变成网页界面，提供对话窗口、侧边栏和文件上传控件。 |
| `langchain` | RAG 编排框架，负责把"文档处理 → 检索 → 提示词组装 → 调用模型"串成一条流水线。 |
| `langchain-text-splitters` | 文档切分器，把长文档按标题层级切成大小合适、语义完整的文本块。 |
| `langchain-openai` | OpenAI 接口的 LangChain 客户端，用它连 LM Studio 的本地模型服务。 |
| `langchain-chroma` | LangChain 与 ChromaDB 之间的桥接层，让向量库能直接参与链路编排。 |
| `chromadb` | 本地文件型向量数据库，负责存储和按语义相似度找回相关段落。 |
| `pypdf` | 解析 .pdf 文件，提取其中的文字内容。 |
| `python-dotenv` | 从 `.env` 文件读取配置和密钥，避免把敏感信息写死在代码里。 |

`.md`、`.txt` 由 Python 标准库直接读取；`.docx` 也是标准库解析（docx 文件本质是
zip 包加 xml，无需第三方库）。因此依赖清单里没有文档解析类的重型库。

---

## 三、环境准备

本项目使用本机的 **Python 3.14.4**（已实测全部依赖均可正常安装与导入）。

```bash
# 1) 创建虚拟环境（项目初始化时已自动创建，此命令供重装时参考）
py -3.14 -m venv .venv

# 2) 安装依赖（国内网络建议加清华镜像，安装更快）
.venv/Scripts/pip install -r requirements.txt
# 若卡住，改用：
.venv/Scripts/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 准备配置文件
cp .env.example .env
#    然后打开 .env，把 CHAT_MODEL / EMBEDDING_MODEL 填成 LM Studio 中已加载的模型名
```

LM Studio 侧需要做的唯一一件事：打开软件 → 加载一个对话模型和一个 embedding 模型 → 启动
**Local Server**（默认 `http://localhost:1234/v1`）。不需要 Docker、Redis 或数据库服务。

---

## 四、使用方式

### 4.1 把文档灌进知识库（第二步）

先确认 LM Studio 已加载好向量模型（默认 `bge-m3`）并启动了 Local Server，然后：

```bash
# 正式入库：解析 → 切分 → 向量化 → 写入 ChromaDB
.venv/Scripts/python.exe -m src.ingest

# 只看解析与切分结果，不连接模型（不需要 LM Studio）
.venv/Scripts/python.exe -m src.ingest --dry-run

# 查看库里现有数据和内容抽样
.venv/Scripts/python.exe -m src.ingest --stats
```

默认是**全量重建**（先清空再写入），避免重复入库；加 `--keep` 则改为追加。

### 4.2 启动完整应用（第四步完成后）

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，在输入框里提问即可。

---

## 五、开发进度

- [x] 第一步：项目初始化（目录结构、依赖清单、说明文档）
- [x] 第二步：数据加载与切分，向量化入库 —— 14 个文档 → 72 个文本块，已用 `text-embedding-bge-m3` 完成入库
- [ ] 第三步：检索与生成，带来源标注
- [ ] 第四步：Streamlit 前端 UI

---

## 六、已知问题（待验证）

- PDF 和 DOCX 的第一块为纯标题块（约 7 个），无实质正文内容。
  - 待验证：是否会影响检索精度（可能被误召回）。
  - 计划：第三步检索测试时观察，如有影响则过滤 title-only chunk。
