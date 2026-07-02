# Knowledge Agent

知识沉淀 Agent — 多源采集 + LLM 结构化抽取 + 混合存储（向量 + 图谱） + 多 Agent 协作 + 持久化记忆。

## 安装

```bash
poetry install
```

## 配置

通过环境变量或 `.env` 文件配置（前缀 `KA_`）：

```bash
# LLM
export KA_OPENAI_API_KEY="sk-xxx"
export KA_OPENAI_BASE_URL="https://api.openai.com/v1"
export KA_LLM_MODEL="gpt-4o"

# Embedding
export KA_EMBEDDING_MODEL="text-embedding-3-small"
```

## 使用

```bash
# 摄入文档
ka ingest ./docs/

# 提问
ka query "什么是 GraphRAG？"

# 启动 API 服务
ka serve
```

## API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/ingest` | POST | 上传并摄入文档文件 |
| `/query` | POST | RAG 问答 |
| `/documents` | GET | 列出已摄入文档 |
| `/health` | GET | 健康检查 |

## 架构

```
用户交互层 (CLI / REST API)
    ↓
Agent 编排层 (QAAgent)
    ↓
检索层 (HybridRetriever: 向量 + BM25 + RRF 融合)
    ↓
存储层 (ChromaDB 向量存储 + SQLite 文档元数据)
    ↓
数据处理层 (Loader → Chunker → Embedder)
```

## 开发计划

详见 [PLAN.md](PLAN.md) 和 [DESIGN.md](DESIGN.md)。
