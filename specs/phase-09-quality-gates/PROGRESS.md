# Phase 9 Progress

## 状态

实现完成，PR #23 pytest 回归已修复，等待远端 CI 验证；完整依赖验收暂缓

## 记录

### 2026-07-31

- [x] 同步 `codex/phase9-quality-gates` 远端分支并复核 PR #23 状态。
- [x] 确认 Python 3.11/3.12 的失败点均为 `Run Ruff`，Node.js 20 提示仅为 warning。
- [x] 使用 CI 当前安装的 Ruff 0.16.1 复现 152 项检查错误。
- [x] 确认显式沿用项目既有的 `E4`、`E7`、`E9`、`F` 规则集后检查通过。
- [x] 将 Ruff 开发依赖、pre-commit 和 CI 版本统一到 0.16.x，并限制 CI 不跨次版本漂移。
- [x] 将 `actions/checkout`、`actions/setup-python` 升级到 Node.js 24 运行时版本。
- [x] 在 CI 日志中输出 pytest 和 Ruff 版本，便于后续诊断。
- [x] Ruff 0.16.1 在 Python 3.12 隔离环境检查通过。
- [x] 首次修复提交后确认远端 Ruff 已通过，后续失败进入 `Run tests`。
- [x] 为 GitHub Actions 添加 pytest 失败注解，公开展示失败测试和断言位置。
- [x] 修正 API reranker 测试的 mock 路径，使其匹配 `openai.OpenAI` 延迟导入。
- [x] 完善 HTML loader 无 BeautifulSoup 时的降级清洗，整块移除 script/style/nav/footer/header 内容。
- [x] 保留空 JSON 数组和对象的有效文档表示，避免被误判为空内容。
- [ ] 确认上述三个 pytest 回归在 PR #23 的 Python 3.11/3.12 CI 中通过。

### 2026-07-30

- [x] 将本地 `main` 从 `004ba12` 快进到远端 `856e020`。
- [x] 检查 ROADMAP、核心入口、测试计划和静态检查结果。
- [x] 确认 API、CLI、Web UI 监控存在合并回归。
- [x] 建立 `specs/` 目录和后续 Phase 索引。
- [x] 修复 API app factory 和评估路由。
- [x] 修复 CLI 命令注册、参数和评估数据集导入。
- [x] 恢复 Orchestrator/Web UI 监控并保留缓存能力。
- [x] 移除 tokenizer 的模块导入时网络访问。
- [x] 补充和修正 API、CLI、Web UI、Orchestrator 回归测试。
- [x] 将依赖真实 LLM/存储的“端到端测试”改为显式依赖注入。
- [x] 添加 Python 3.11/3.12 GitHub Actions CI。
- [x] 更新 README/ROADMAP。
- [x] `python3 -m compileall -q knowledge_agent tests` 通过。
- [x] `.venv/bin/python -m ruff check knowledge_agent tests` 通过。
- [x] 可用依赖范围内 46 项轻量测试通过。
- [ ] 在 Python 3.12 完整依赖环境运行全量 pytest（按用户要求暂缓）。

## 已知基线问题

- 仓库原 `.venv` 使用 Python 3.14.6，且缺少 `openai`、`chromadb` 等运行时依赖，
  不能作为项目声明的 Python 3.11/3.12 验证环境。
- 已在 `/tmp/knowledge-agent-phase9` 创建 Python 3.12 临时环境，但 PyPI 下载速度
  不稳定；按用户要求，本轮已停止依赖安装和全量 pytest。
- tokenizer 已加入字符级离线降级，并有回归测试覆盖。
- PR #23 的远端 CI 会在修复提交推送后重新执行；结果以 GitHub Actions 为准。

## 后续验收

- 在网络稳定时安装完整依赖并执行 `python -m pytest -q`。
- 由 GitHub Actions 在 Python 3.11/3.12 环境持续执行编译、Ruff 和 pytest。
