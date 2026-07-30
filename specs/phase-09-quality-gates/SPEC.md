# Phase 9 Spec: 主线稳定性与 CI 质量门禁

## 背景

同步 `main` 到 `856e020` 后，主线已经包含监控、缓存、搜索增强、知识库管理和
多模态能力，但合并过程中出现了入口点回归：

- API 模块在 `app` 创建前注册评估路由，导入会失败。
- CLI 的 `delete` 命令装饰器和评估数据集导入丢失。
- Web UI 仍绑定监控回调，但对应函数和 Orchestrator 监控接口丢失。
- 测试未覆盖这些入口点，且部分“端到端测试”实际依赖真实服务。
- 当前没有远端 CI，静态检查和测试无法形成合并门禁。
- tokenizer 在模块导入时可能访问网络，离线测试不稳定。

## 目标

1. 恢复 API、CLI、Web UI 入口点的可导入和可调用状态。
2. 恢复监控与缓存能力的共存，避免功能合并互相覆盖。
3. 让 tokenizer 初始化不再产生模块导入时的网络访问。
4. 补充覆盖入口点和监控行为的回归测试。
5. 添加 GitHub Actions，在 push 和 pull request 时执行静态检查与测试。
6. 校准 README/ROADMAP，使完成状态与当前代码一致。

## 非目标

- 不引入 Docker 或发布镜像。
- 不迁移到 LangGraph。
- 不替换 ChromaDB、SQLite 或 NetworkX。
- 不新增业务功能或重做 Web UI 视觉设计。
- 不要求本阶段解决所有可选依赖的生产部署问题。

## 实现要求

### API

- 所有路由必须在 `create_app()` 创建的 FastAPI 实例上注册。
- `/evaluate/retrieval` 和 `/evaluate/answer` 必须出现在路由表中。
- 响应模型的列表字段使用 `default_factory`。
- 上传接口所需的 `python-multipart` 必须声明为运行时依赖。

### CLI

- `delete` 必须作为顶级 CLI 命令注册。
- `eval-dataset` 的子命令必须在调用时正确导入 `EvaluationDataset`。
- `ingest` 恢复 `--chunk-size` 和 `--chunk-overlap`，并将配置传给
  `RecursiveChunker`/`CollectionAgent`。

### 监控与 Web UI

- Orchestrator 同时初始化 `MetricsCollector`、`Tracer` 和 `QueryCache`。
- 普通查询、流式查询、缓存命中和异常必须产生对应计数或耗时数据。
- Web UI 恢复监控报告和重置回调。
- 缓存统计读取当前 Orchestrator 的缓存，而不是新建空缓存实例。

### 测试与质量门禁

- 新增 API app factory 路由测试。
- 扩充 CLI 命令和参数测试。
- 修正不隔离外部依赖的测试，确保测试默认不访问 LLM、模型下载或持久化服务。
- Ruff 检查通过。
- GitHub Actions 使用受支持的 Python 版本执行 Ruff 和 pytest。

## 验收标准

- `python -m compileall -q knowledge_agent tests` 通过。
- `python -m ruff check knowledge_agent tests` 通过。
- `python -m pytest -q` 通过，且不需要 API Key。
- `create_app()` 可成功构建，关键 API 路由完整。
- CLI 命令表包含 `ingest`、`query`、`delete`、`eval`、`eval-dataset`、
  `serve` 和 `webui`。
- `.github/workflows/ci.yml` 存在并覆盖 lint/test。
- `PROGRESS.md` 记录最终验证结果和剩余风险。
