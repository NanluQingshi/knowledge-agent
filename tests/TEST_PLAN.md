# Test Coverage — Issue Tracker

> 当前状态: 15 个测试文件, ~365 个测试用例  
> 目标: 核心模块全覆盖

---

## 优先级 1 — 缺失独立测试的核心模块

### TC-01: `cli.py` CLI 命令
- [x] `cli` 模块导入
- [x] 4 个命令存在 (ingest/query/serve/webui)
- [x] 各命令参数验证

### TC-02: `config.py` 配置管理
- [x] 默认值正确
- [x] 环境变量覆盖 (OPENAI_API_KEY, LLM_MODEL, CHUNK_SIZE, RETRIEVAL_TOP_K)
- [x] 多变量同时覆盖

---

## 优先级 2 — 增强已有测试覆盖

### TC-03: `exporter.py` 导出
- [x] `export_json` 输出有效 JSON
- [x] `export_markdown` 输出有效 Markdown + _index.md
- [x] 空知识库导出不崩溃

### TC-04: `enhancer.py` 更多用例
- [x] `MultiQueryFusion` 去重、RRF 评分、top_k
- [ ] `QueryRewriter` API 失败时优雅降级
- [ ] `HyDEGenerator` API 失败时返回原问题

---

## 优先级 3 — 集成测试

### TC-05: 端到端流程
- [ ] 摄入 → 检索 → 问答 路径不崩溃（mock LLM）
- [ ] 缓存 → 命中 → 返回 路径正确
