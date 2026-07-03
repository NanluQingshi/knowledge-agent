# 知识沉淀 Agent 执行计划

## 总体策略

按照设计文档建议的「渐进式架构演进」策略，从 MVP 到完整系统分 5 个 Phase 迭代构建。每个 Phase 产出可独立运行的中间版本。

---

## Phase 1：MVP 核心 — 文档摄入 + RAG 问答

**目标**：实现「存文档 + 语义搜索 + 基础问答」的最小闭环。

### 1.1 项目脚手架
- 技术选型：Python 3.11+, Poetry 管理依赖
- 目录结构：
  ```
  knowledge_agent/
  ├── cli.py                     # CLI 入口
  ├── config.py                  # 配置管理
  ├── loaders/                   # 文档加载器
  │   ├── __init__.py
  │   ├── base.py                # 抽象基类
  │   ├── text_loader.py         # 纯文本
  │   ├── markdown_loader.py     # Markdown
  │   └── pdf_loader.py          # PDF
  ├── chunkers/                  # 分块策略
  │   ├── __init__.py
  │   ├── base.py
  │   ├── fixed_chunker.py       # 固定大小分块
  │   ├── semantic_chunker.py    # 语义分块
  │   └── recursive_chunker.py   # 递归分块
  ├── embeddings/                # 向量化
  │   ├── __init__.py
  │   └── embedder.py            # OpenAI/本地 embedding
  ├── storage/                   # 存储层
  │   ├── __init__.py
  │   ├── vector_store.py        # ChromaDB 向量存储
  │   └── doc_store.py           # 文档元数据存储
  ├── retrieval/                 # 检索
  │   ├── __init__.py
  │   ├── vector_retriever.py    # 向量检索
  │   └── bm25_retriever.py      # BM25 关键词检索
  ├── agents/                    # Agent 层
  │   ├── __init__.py
  │   └── qa_agent.py            # RAG 问答 Agent
  └── api/                       # API 层
      ├── __init__.py
      └── routes.py              # FastAPI 路由
  ```

### 1.2 核心模块
| 模块 | 核心类/函数 | 职责 |
|------|------------|------|
| loaders | `BaseLoader`, `TextLoader`, `MarkdownLoader`, `PDFLoader` | 多格式文档加载 |
| chunkers | `FixedChunker`, `SemanticChunker`, `RecursiveChunker` | 文档分块 |
| embeddings | `Embedder` (OpenAI compatible) | 文本向量化 |
| storage | `VectorStore` (ChromaDB), `DocStore` (SQLite) | 向量+元数据存储 |
| retrieval | `VectorRetriever`, `BM25Retriever`, `HybridRetriever` | 多路检索 |
| agents | `QAAgent` | RAG 问答，含上下文构建和 LLM 调用 |
| api | FastAPI routes: `/upload`, `/query`, `/documents` | REST API |
| cli | `ingest`, `query`, `serve` 子命令 | 命令行交互 |

### 1.3 依赖
```
chromadb, langchain-text-splitters, openai, fastapi, uvicorn,
pypdf, unstructured, tiktoken, rank-bm25, sentence-transformers
```

---

## Phase 2：知识图谱集成

**目标**：引入知识图谱，支持实体/关系抽取和图谱增强检索。

### 2.1 新增模块
```
knowledge_agent/
├── extraction/                  # 知识抽取
│   ├── __init__.py
│   ├── entity_extractor.py      # NER 实体识别
│   ├── relation_extractor.py    # 关系抽取
│   └── triple_extractor.py      # SPO 三元组抽取
├── graph/                       # 图谱层
│   ├── __init__.py
│   ├── graph_store.py           # Neo4j / NetworkX 图谱存储
│   ├── community_detector.py    # Leiden 社区检测
│   └── graph_retriever.py       # 图谱检索
└── retrieval/
    └── graphrag_retriever.py    # GraphRAG 检索（Local + Global Search）
```

### 2.2 核心功能
| 功能 | 描述 |
|------|------|
| LLM 驱动的实体识别 | 从文档中提取人、组织、技术、概念等实体 |
| 关系抽取 | 识别实体间关系（使用、依赖、属于等） |
| 图谱构建 | 实体→节点，关系→边，持久化存储 |
| 社区检测 | Leiden 算法检测知识社区 |
| Local Search | 检索特定实体及邻域上下文 |
| Global Search | 利用社区摘要进行全局问答 |

---

## Phase 3：多 Agent 协作架构

**目标**：用 LangGraph 实现 4 个 Agent 的协作工作流。

### 3.1 Agent 定义
| Agent | 职责 |
|-------|------|
| CollectionAgent | 监听数据源、解析文档、数据清洗 |
| ExtractionAgent | NER + 关系抽取 + 摘要生成 |
| QAAgent | 路由检索策略、生成回答、追问 |
| QualityAgent | 过期检测、冲突发现、版本管理 |

### 3.2 工作流
```
Orchestrator (LangGraph StateGraph)
  ├── Ingest Node → CollectionAgent
  ├── Process Node → ExtractionAgent  
  ├── Query Node → QAAgent
  └── Maintain Node → QualityAgent
```

---

## Phase 4：Agent 记忆系统

**目标**：实现四层记忆模型，使 Agent 具备持续学习能力。

### 4.1 记忆实现
| 记忆类型 | 技术方案 | 存储内容 |
|---------|---------|---------|
| Working Memory | LLM Context Window | 当前对话上下文 |
| Episodic Memory | 向量数据库 + 时间戳 | 历史对话、操作日志 |
| Semantic Memory | 知识图谱 | 提炼后的事实、规则 |
| Procedural Memory | YAML/JSON 配置 | 工作流模板、最佳实践 |

### 4.2 新增模块
```
knowledge_agent/
├── memory/
│   ├── __init__.py
│   ├── working_memory.py
│   ├── episodic_memory.py
│   ├── semantic_memory.py
│   └── procedural_memory.py
```

---

## Phase 5：反馈闭环与持续进化

**目标**：用户反馈、知识质量管理、自动更新机制。

### 5.1 功能清单
| 功能 | 描述 |
|------|------|
| 用户反馈收集 | 有用/无用标记，反馈评分 |
| 知识质量评分 | 基于反馈 + 引用频率的评分算法 |
| 知识新鲜度管理 | 时间衰减 + 动态权重调整 |
| 冲突检测 | 新知识入库自动比对，发现矛盾 |
| 知识缺口发现 | Agent 主动识别缺失领域 |
| 增量更新 | 版本化管理，支持回滚 |

---

## 里程碑与优先级

| Phase | 产出物 | 优先级 | 预估核心文件数 |
|-------|--------|--------|--------------|
| P1 | CLI + API + RAG 问答 | P0 最高 | ~20 |
| P2 | 知识图谱 + GraphRAG | P1 高 | ~10 |
| P3 | 多 Agent 工作流 | P1 高 | ~8 |
| P4 | 四层记忆系统 | P2 中 | ~6 |
| P5 | 反馈闭环 | P2 中 | ~5 |

---

## 技术选型确认

| 层面 | 选型 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | AI/ML 生态最完善 |
| 包管理 | Poetry | 现代 Python 依赖管理 |
| 编排框架 | LangGraph | 多 Agent 工作流、生态丰富 |
| 向量存储 | ChromaDB (MVP) → Milvus (生产) | ChromaDB 零配置快速起步 |
| 图谱存储 | NetworkX (MVP) → Neo4j (生产) | NetworkX 无需外部服务 |
| LLM | OpenAI API (兼容 Anthropic) | 多模型适配 |
| 文档解析 | unstructured + PyPDF | 多格式支持 |
| Embedding | sentence-transformers / OpenAI | 本地 + 云端可选 |
| API 框架 | FastAPI | 高性能、自动文档 |
| 评估 | RAGAS | RAG 系统标准评估 |

---

## MVP 最小可交付定义

- [ ] 从目录加载 Markdown/PDF/文本文件
- [ ] 自动分块并生成向量
- [ ] 向量语义搜索
- [ ] RAG 问答（引用来源）
- [ ] CLI 工具：`ka ingest` / `ka query` / `ka serve`
- [ ] REST API：POST `/query` / POST `/ingest`
