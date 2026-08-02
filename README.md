# Knowledge Agent

知识沉淀 Agent — 多源采集 + LLM 结构化抽取 + 混合存储（向量 + 图谱）+ 多 Agent 协作 + 持久化记忆 + 可观测监控。

## 安装

```bash
poetry install

# 可选：额外文档格式支持
poetry install -E loaders

# 可选：评估模块（RAGAS）
poetry install -E eval

# 全部
poetry install -E all

# 激活 pre-commit 自动代码检查
pre-commit install
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

# 启动 Gradio Web UI（推荐）
ka webui
```

## Docker 部署

```bash
# 创建本地配置并填写需要的 API Key
cp .env.example .env

# 构建并启动 API 与 Web UI
docker compose up -d

# 也可以只启动其中一种模式
docker compose up -d api
docker compose up -d webui

# 访问 Web UI: http://localhost:7860
# 访问 API: http://localhost:8000

# 查看服务健康状态
docker compose ps

# 查看日志
docker compose logs -f
```

默认镜像以非 root 用户运行、安装常用文档加载器，并使用具名卷
`knowledge-agent-data` 持久化数据。可在 `.env` 中修改宿主机端口、卷名和镜像标签。

OCR 系统依赖默认不安装，需要图片文字识别时执行：

```bash
KA_DOCKER_INSTALL_OCR=true docker compose build
docker compose up -d
```

评估模块等全部可选 Python 依赖可通过 `KA_DOCKER_EXTRAS=all` 构建。运行时密钥只从
Compose 环境变量注入，`.env` 和本地 `data/` 不会进入镜像构建上下文。

## 功能模块

### Web UI（Gradio）

```bash
ka webui
```

启动后打开 http://localhost:7860，支持以下功能 Tab：

| Tab | 功能 |
|-----|------|
| 📥 摄入文档 | 拖拽上传文件 / 输入 URL 抓取网页 |
| 💬 智能问答 | 流式 RAG 问答，支持多轮对话和跨会话记忆 |
| 🕸️ 知识图谱 | 交互式 pyvis 实体关系图可视化 |
| 📚 文档列表 | 查看/版本历史/删除文档 |
| 📊 系统状态 | 知识库健康报告、新鲜度分布、记忆统计 |
| 📈 监控 | 实时性能指标（P50/P95/P99 延迟、请求计数） |
| 📋 评估 | 检索质量（MRR/Recall/NDCG）和答案质量评估 |
| ⚙️ 设置 | OpenAI / Anthropic API Key 动态配置 |
| 📤 导出 | 将知识库导出为 Markdown 或 JSON |
| 🏷️ 标签管理 | 搜索文档并维护标签 |

### CLI

| 命令 | 说明 |
|------|------|
| `ka ingest <path>` | 摄入文档，可配置 chunk 大小、抽取和质检 |
| `ka query <question>` | RAG 问答 |
| `ka delete <doc-id>` | 删除文档及关联数据 |
| `ka eval [retrieval\|answer\|all]` | 运行检索或答案质量评估 |
| `ka eval-dataset <command>` | 管理评估数据集 |
| `ka serve` | 启动 REST API 服务 |
| `ka webui` | 启动 Gradio Web UI |

### REST API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/ingest` | POST | 上传并摄入文档文件 |
| `/query` | POST | RAG 问答 |
| `/query/stream` | POST | Server-Sent Events 流式问答 |
| `/documents` | GET | 列出已摄入文档 |
| `/documents/{doc_id}` | DELETE | 删除文档及关联数据 |
| `/evaluate/retrieval` | POST | 评估检索质量 |
| `/evaluate/answer` | POST | 评估答案质量 |
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
| URL 网页 | UrlLoader | `httpx` + `beautifulsoup4` (可选) |

## 核心能力

### 检索

- **混合检索**: 向量语义检索 + BM25 关键词检索 + RRF 融合 + Cross-Encoder 重排序
- **中文分词优化**: BM25 集成 jieba 分词，大幅提升中文检索质量
- **GraphRAG**: 局部实体邻域检索 + 全局社区摘要检索（Local / Global Search）
- **查询缓存**: LRU 缓存相同提问结果（TTL 300s），避免重复计算

### 知识图谱

- LLM 驱动的实体识别（NER）/ 关系抽取 / SPO 三元组提取
- NetworkX 图存储，Louvain 社区检测
- 交互式知识图谱可视化（pyvis）

### Agent 系统

- **多 Agent 协作**: 采集 / 抽取 / 问答 / 质检 四个 Agent 协同工作
- **四层记忆系统**: 工作记忆（对话窗口）、情景记忆（ChromaDB）、语义记忆（图谱）、程序记忆（工作流模板）
- **多轮对话**: 支持上下文延续和跨会话历史记忆检索

### 工程化

- **反馈闭环**: 用户反馈收集、知识质量评分、新鲜度管理
- **评估体系**: 检索质量评估（MRR/Recall/Precision/NDCG）+ RAGAS 答案质量评估
- **文档版本管理**: 自动版本链追踪，支持回滚
- **监控仪表盘**: 结构化 JSON 日志、请求 Trace ID、性能指标（P50/P95/P99）
- **并行摄入**: 多线程批量摄入，加速大目录处理
- **Embedding 回退**: OpenAI API 不可用时自动回退到本地 sentence-transformers 模型
- **API Key 管理**: Web UI 运行时动态配置 LLM 提供商
- **pre-commit hooks**: ruff 自动化代码格式与 lint 检查
- **CI 质量门禁**: GitHub Actions 在 Python 3.11/3.12 执行编译、Ruff 和 pytest

## 项目文档

- [PLAN.md](PLAN.md) — 原始 5 个 Phase 开发计划
- [ROADMAP.md](ROADMAP.md) — 后续迭代路线图与完成状态
- [DESIGN.md](DESIGN.md) — 设计方案与技术选型报告
- [specs/README.md](specs/README.md) — 后续实现 spec 索引与进度目录
