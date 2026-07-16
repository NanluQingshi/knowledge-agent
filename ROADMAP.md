# 知识沉淀 Agent — 路线图与完成状态

> 记录项目所有迭代阶段、已完成功能和待办事项。  
> 每个阶段对应一个或多个 git 分支，独立开发、合并到 main。

---

## 状态总览

| 阶段 | 主题 | 分支 | 状态 |
|------|------|------|------|
| Phase 1-5 | 原始开发计划（MLP → 完整系统） | `feature/implementation` | ✅ 已合并 |
| Phase 6 | 修复与优化（CR-01 ~ NF-02） | `feature/phase6-fixes` | ✅ 已合并 |
| Phase 7 | 功能增强 | 多个分支 | ✅ 已合并 |
| Phase 8 | 监控与性能 | `feature/monitoring`、`feature/performance` | 🚀 待合并 |
| 待办 | 下一步方向 | — | 📋 规划中 |

---

## Phase 1-5: 原始开发计划

PLAN.md 中定义的 5 个 Phase，从 MVP 到完整系统。

| Phase | 内容 | 状态 |
|-------|------|------|
| P1 | CLI + API + RAG 问答 | ✅ 已完成 |
| P2 | 知识图谱 + GraphRAG | ✅ 已完成 |
| P3 | 多 Agent 工作流 | ✅ 已完成 |
| P4 | 四层记忆系统 | ✅ 已完成 |
| P5 | 反馈闭环 | ✅ 已完成 |

**分支**: `feature/implementation` | **Commits**: 多个 | **状态**: ✅ 已合并到 main

---

## Phase 6: 修复与优化

基于项目评估报告识别的问题修复。

| 批次 | 条目 | 状态 |
|------|------|------|
| 🔴 Critical | CR-01 向量维度一致 / CR-02 BM25 优化 / CR-03 客户端复用 | ✅ 已完成 |
| 🟡 Major | MJ-01~MJ-05 文档/反馈/重构/依赖/文件锁 | ✅ 已完成 |
| 🟢 Minor | MN-01~MN-04 时间戳/docstring/代码抽取 | ✅ 已完成 |
| 📋 Feature | NF-01 Embedder 回退 / NF-02 维度日志 | ✅ 已完成 |

**分支**: `feature/phase6-fixes` | **Commits**: 11 个 | **状态**: ✅ 已合并到 main

---

## Phase 7: 功能增强

### 7.1 对话历史

| 功能 | 说明 |
|------|------|
| Orchestrator 透传 chat_history | `run_query()` / `run_query_stream()` 新增参数 |
| Web UI 多轮对话 | Gradio history 传入 LLM，同一会话连贯问答 |
| 跨会话记忆检索 | 新问题前自动检索相关历史记忆注入上下文 |
| 公共记忆 API | `orchestrator.recall_memories()` / `get_memory_stats()` |
| 清空历史按钮 | Web UI 问答 Tab 下方 |

**分支**: `feature/chat-history` | **Commits**: 5 个 | **状态**: ✅ 已合并到 main

### 7.2 打磨与完善

| 功能 | 说明 |
|------|------|
| 封装 EvaluationRunner 私有访问 | 改为使用公共 `Orchestrator.retrieve()` |
| EvaluationDataset 文件锁 | `fcntl` 保护 JSON 读写 |
| CSVLoader 异常处理 | 窄化异常捕获范围 + 日志 |
| Reranker 清理 | 移除无用 `settings` touch 语句 |
| 中文分词优化 | BM25 集成 jieba 分词 |
| URL 网页抓取 | UrlLoader + Web UI URL 输入 |
| 知识图谱可视化 | pyvis 交互式实体关系图 |
| 测试覆盖 | URL Loader + Web UI 冒烟测试 |
| pre-commit hooks | ruff 自动化格式 + lint |
| 文档版本管理 | 版本链追踪 + 回滚 |
| API Key 管理 | Web UI 设置 Tab 动态配置 |
| README 更新 | 同步所有新功能 |

**分支**: `feature/polish` | **Commits**: 10 个 | **状态**: ✅ 已合并到 main

---

## Phase 8: 监控与性能

### 8.1 监控与可观测性

| 功能 | 文件 | 说明 |
|------|------|------|
| 结构化日志 | `monitoring/logger.py` | JSON 格式，含时间戳/级别/trace_id |
| 性能指标 | `monitoring/metrics.py` | P50/P95/P99 延迟统计 + 计数器 |
| 请求追踪 | `monitoring/tracer.py` | 每个请求唯一 Trace ID，ContextVar 传递 |
| 监控面板 | Web UI 📈 监控 Tab | 实时显示操作耗时和请求计数，支持重置 |
| CLI 集成 | `cli.py` | 启动时初始化结构化日志 |

**分支**: `feature/monitoring` | **Commits**: 1 个 | **状态**: 🚀 待合并到 main

### 8.2 性能优化与缓存

| 功能 | 文件 | 说明 |
|------|------|------|
| 查询结果缓存 | `cache.py` | LRU 淘汰 + TTL 过期，重复提问直接返回 |
| 并行批量摄入 | `collection_agent.py` | `ingest_path_parallel()` 多线程并发处理 |
| 缓存统计 | Web UI | 缓存命中条数查询 |

**分支**: `feature/performance` | **Commits**: 1 个 | **状态**: 🚀 待合并到 main

---

## 待办事项（规划中）

| 优先级 | 方向 | 说明 |
|--------|------|------|
| ⭐ | CI/CD 自动化 | GitHub Actions：ruff check + pytest + 自动发布 |
| ⭐ | Docker 部署 | Dockerfile + docker-compose，一键启动 |
| | 搜索增强 | Query Rewriting、HyDE 检索、多查询融合 |
| | 知识库搜索/浏览 | 全文搜索、标签/分类过滤 |
| | 知识库导出 | 导出为 Markdown / JSON / CSV |
| | 多模态支持 | 图片理解、OCR |

---

## 分支索引

| 分支 | 主题 | Commits | 状态 |
|------|------|---------|------|
| `main` | 主线 | — | 最新发布 |
| `feature/implementation` | Phase 1-5 原始开发 | 多个 | ✅ 已合并 |
| `feature/phase6-fixes` | Phase 6 修复 | 11 | ✅ 已合并 |
| `feature/chat-history` | 对话历史 | 5 | ✅ 已合并 |
| `feature/web-ui` | Web UI | 1 | ✅ 已合并 |
| `feature/polish` | 打磨完善 | 10 | ✅ 已合并 |
| `feature/monitoring` | 监控与可观测性 | 1 | 🚀 待合并 |
| `feature/performance` | 性能优化与缓存 | 1 | 🚀 待合并 |