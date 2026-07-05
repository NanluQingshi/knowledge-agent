# Phase 6: 修复与优化路线图

> 补充 [PLAN.md](PLAN.md) 中 5 个 Phase 之后的修复与增强计划。  
> 每个条目对应一个独立的 git commit，按优先级分批执行。

---

## 状态总览

| 批次 | 条目 | 文件 | 状态 |
|------|------|------|------|
| 批 1 — Critical | 3 项 | `vector_store.py`, `cli.py`, `routes.py`, `episodic_memory.py`, `orchestrator.py` | ✅ 已完成 |
| 批 2 — Major | 5 项 | `orchestrator.py`, `PLAN.md`, `collector.py`, `scorer.py`, `loaders/__init__.py`, `cli.py`, `routes.py`, `pyproject.toml` | ✅ 已完成 |
| 批 3 — Minor | 4 项 | `quality_agent.py`, `freshness.py`, `base.py`, `recursive_chunker.py`, `semantic_chunker.py` | ✅ 已完成 |
| 批 4 — Feature | 2 项 | `embedder.py`, `vector_store.py` | ✅ 已完成 |

---

## 批 1 🔴 Critical

### CR-01 移除 VectorStore 内置 embedding function
- **问题**: `VectorStore` 使用 `all-MiniLM-L6-v2` (384 维) 作为 ChromaDB 内置 embedding function，但系统中 `Embedder` 默认使用 OpenAI `text-embedding-3-small` (1536 维)。虽然 `add()` 传入了显式 embeddings，但 Collection 的内置函数与实际数据维度不匹配。
- **方案**: 移除 `embedding_function` 参数，不设内置函数。所有 embedding 完全由外部 `Embedder` 管理。
- **commit**: `b81c38f`

### CR-02 优化 BM25 索引构建
- **问题**: 每次查询为构建 BM25 索引都执行 `vector_store.search(top_k=count())` 全量向量扫描，文档量增大后性能急剧恶化。
- **方案**: 新增 `VectorStore.get_all_documents()` 方法直接读取存储数据（零向量计算），CLI 和 API 路由均改用此方法。
- **commit**: `040e833`

### CR-03 复用 VectorStore 的 ChromaDB 客户端
- **问题**: `EpisodicMemory` 创建独立的 `PersistentClient` 实例和内置 embedding function，与系统共用的 `VectorStore` 写入同一持久化目录，可能导致数据竞争。
- **方案**: 通过 `VectorStore.chroma_client` 属性暴露统一客户端，`EpisodicMemory` 复用该客户端创建独立 collection。
- **commit**: `016b46a`

---

## 批 2 🟡 Major

### MJ-01 更新 LangGraph 文档状态
- **问题**: PLAN.md 描述使用 LangGraph StateGraph，但实际实现为手动顺序编排。
- **方案**: 更新 PLAN.md 中 Phase 3 描述和技术选型表格，反映实际决策。同时修复 `Orchestrator._get_qa_agent()` 中残留的 BM25 全量扫描。
- **commit**: `1f5f955`

### MJ-02 按文档粒度的反馈评分
- **问题**: `KnowledgeScorer._get_usefulness_for_doc(doc_id)` 始终返回全局有用率，忽略了 doc_id。
- **方案**: 新增 `FeedbackCollector.get_stats_for_doc(doc_id)`，在 `source_doc_ids` JSON 字段中搜索匹配记录。`KnowledgeScorer` 优先使用文档级统计，无记录时回退全局统计。
- **commit**: `b5335f5`

### MJ-03 提取重复的 `_all_loaders()`
- **问题**: `cli.py` 和 `routes.py` 各自定义了完全相同的 `_all_loaders()` 函数。
- **方案**: 抽取到 `knowledge_agent/loaders/__init__.py` 作为公共函数 `all_loaders()`，两处统一导入并清理不再直接使用的加载器导入。
- **commit**: `e77fbd4`

### MJ-04 移除未使用的依赖
- **问题**: `pyproject.toml` 声明了 `langchain-text-splitters`，但代码中未使用。
- **方案**: 从 `[tool.poetry.dependencies]` 中移除。
- **commit**: `f62ba23`

### MJ-05 文件锁保护 JSON 持久化
- **问题**: `ProceduralMemory` 的 `_load()` / `_save()` 无并发保护，多进程场景可能数据丢失。
- **方案**: 使用 `fcntl.flock` 加共享锁 (`LOCK_SH`) 读、排他锁 (`LOCK_EX`) 写。
- **commit**: `4c8822a`

---

## 批 3 🟢 Minor

### MN-01 QualityAgent 时间戳比较
- **问题**: `check_expired_documents()` 使用字符串 `<` 比较 ISO 时间戳。
- **方案**: 解析为 `datetime` 对象后再比较，遇到无法解析的时间戳则跳过该文档。
- **commit**: `1891055` (与本批其他条目合并)

### MN-02 FreshnessManager 评分范围说明
- **问题**: docstring 写 `0.0 ~ 1.0+` 但未说明何时会超过 1.0。
- **方案**: 补充注释说明引用次数加成 (1+log(1+refs)) 可能导致评分超过 1.0，调用方可按需 clamp。
- **commit**: `1891055` (与本批其他条目合并)

### MN-03 提取 `_count_tokens` 到基类
- **问题**: `RecursiveChunker` 和 `SemanticChunker` 各自定义了相同的 `_count_tokens` 方法。
- **方案**: 添加到 `BaseChunker` 作为静态方法 `count_tokens()`，子类通过 `self.count_tokens()` 调用。
- **commit**: `1891055` (与本批其他条目合并)

### MN-04 SemanticChunker overlap 单位说明
- **docstring 已明确** overlap 以**句子数**为单位，与 `chunk_size`（token 数）不同。本次仅做文档确认，无代码变更。
- **commit**: `1891055` (与本批其他条目合并)

---

## 批 4 📋 New Features

### NF-01 Embedder 本地回退
- **问题**: 当 `openai_api_key` 为空时，`Embedder` 在 API 调用时才会失败。
- **方案**: 构造时检查 API key 是否存在；不存在时跳过 OpenAI 客户端初始化。`embed()` 方法先尝试 API，失败时自动回退到 `sentence-transformers` 本地模型 (`all-MiniLM-L6-v2`)。
- **commit**: `a1b0fc2`

### NF-02 Embedding 维度日志
- **问题**: 缺少 embedding 维度的可观测性，维度不匹配时难以排查。
- **方案**: 在 `VectorStore.add()` 中添加 `logger.debug` 输出向量维度。
- **commit**: `7f52e3c`

---

## 分支信息

- **分支**: `feature/phase6-fixes`
- **基分支**: `feature/tests-and-deps`
- **Commits**: 11 个
- **状态**: ✅ 全部完成
