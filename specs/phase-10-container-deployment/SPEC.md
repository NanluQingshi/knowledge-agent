# Phase 10 Spec: Docker 与本地部署

## 目标

为 API 和 Web UI 提供可重复的一键部署方式，使新环境不需要手动安装 Python
和系统依赖。

## 建议范围

- 多阶段 `Dockerfile`，默认以非 root 用户运行。
- `docker-compose.yml`，持久化 `data/` 并暴露 API/Web UI 端口。
- 健康检查和环境变量示例。
- OCR 等可选系统依赖通过独立构建参数或镜像变体启用。
- 增加容器启动、健康检查和数据持久化验证。

## 验收标准

- `docker compose up` 可启动至少一种服务模式。
- `/health` 在容器内外均可访问。
- 重启容器后知识库数据仍存在。
- 镜像中不包含本地密钥或开发缓存。

## 前置条件

Phase 9 CI 和入口点修复完成。
