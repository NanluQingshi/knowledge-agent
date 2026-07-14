# Knowledge Agent

知识沉淀 Agent — 多源采集 + LLM 结构化抽取 + 混合存储（向量 + 图谱）+ 多 Agent 协作 + 持久化记忆。

## 安装

```bash
poetry install

# 可选：额外文档格式支持
poetry install -E loaders

# 可选：评估模块（RAGAS）
poetry install -E eval

# 全部
poetry install -E all
```

## 配置

通过环境变量或 `.env` 文件配置（前缀 `KA_`）：

```bash
# LLM
export KA_OPENAI_API_KEY="sk-xxx"
export KA_OPENAI_BASE_URL="https://api.openai.com/v1"
export KA_LLM_MODEL="gpt-4o"

# Embedding（不设置 API Key 时自动用本地 sentence-transformers）
export KA_EMBEDDING_MODEL="text-embedding-3-small"
```

## 使用

```bash
# 摄入文档
ka ingest ./docs/

# 提问
ka query "什么是 GraphRAG？"

# 启动 REST API 服务
ka serve

# 启动 Gradio Web UI
ka webui
```

## 功能模块

### Web UI（Gradio）

```bash
ka webui
```

启动后打开 http://localhost:7860，支持：

| Tab | 功能 |
|-----|------|
| 📥 摄入文档 | 拖拽上传 .txt / .md / .pdf / .csv / .json 等文件 |
| 💬 智能问答 | 流式 RAG 问答，支持多轮对话历史和跨会话记忆 |
| 📚 文档列表 | 查看/删除已摄入文档 |
| 📊 系统状态 | 知识库健康报告、新鲜度分布、记忆统计 |

### CLI

| 命令 | 说明 |
|------|------|
| `ka ingest <path>` | 摄入文档（文件或目录） |
| `ka query <question>` | RAG 问答 |
| `ka serve` | 启动 REST API 服务 |
| `ka webui` | 启动 Gradio Web UI |

### REST API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/ingest` | POST | 上传并摄入文档文件 |
| `/query` | POST | RAG 问答 |
| `/documents` | GET | 列出已摄入文档 |
| `/health` | GET | 健康检查 |

## 支持的文档格式

| 格式 | 加载器 | 依赖 |
|------|--------|------|
| `.txt` / `.log` | TextLoader | 内置 |
| `.md` | MarkdownLoader | 内置 |
| `.pdf` | PDFLoader | `pypdf` |
| `.csv` | CSVLoader | 内置 |
| `.json` | JSONLoader | 内置 |
| `.docx` | DocxLoader | `python-docx` (可选) |
| `.html` / `.htm` | HTMLLoader | `beautifulsoup4` (可选) |

## 架构

```
用户交互层 (CLI / REST API / Gradio Web UI)
    ↓
Agent 编排层 (Orchestrator: 采集 → 抽取 → 问答 → 质检)
    ↓
检索层 (HybridRetriever: 向量 + BM25 + Cross-Encoder 重排序 + GraphRAG)
    ↓
存储层 (ChromaDB 向量存储 + SQLite 文档元数据 + NetworkX 知识图谱)
    ↓
数据处理层 (Loader → Chunker → Embedder → NER/关系抽取)
```

### 核心能力

- **混合检索**: 向量语义检索 + BM25 关键词检索 + RRF 融合 + Cross-Encoder 重排序
- **知识图谱**: LLM 驱动的实体识别 / 关系抽取 / SPO 三元组提取，Louvain 社区检测
- **GraphRAG**: 局部实体邻域检索 + 全局社区摘要检索（Local / Global Search）
- **多 Agent 协作**: 采集 / 抽取 / 问答 / 质检 四个 Agent 协同工作
- **四层记忆系统**: 工作记忆（对话窗口）、情景记忆（ChromaDB）、语义记忆（图谱）、程序记忆（工作流模板）
- **反馈闭环**: 用户反馈收集、知识质量评分、新鲜度管理
- **评估体系**: 检索质量评估（MRR/Recall/Precision/NDCG）+ RAGAS 答案质量评估
- **Embedding 回退**: OpenAI API 不可用时自动回退到本地 sentence-transformers 模型

## 开发计划

详见 [PLAN.md](PLAN.md)、[ROADMAP.md](ROADMAP.md) 和 [DESIGN.md](DESIGN.md)。
