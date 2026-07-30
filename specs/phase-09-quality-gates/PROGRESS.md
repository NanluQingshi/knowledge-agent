# Phase 9 Progress

## 状态

实现完成，完整依赖验收暂缓

## 记录

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

## 后续验收

- 在网络稳定时安装完整依赖并执行 `python -m pytest -q`。
- 由 GitHub Actions 在 Python 3.11/3.12 环境持续执行编译、Ruff 和 pytest。
