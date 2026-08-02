# Phase 10 Progress

## 状态

实现完成：静态结构检查通过，容器运行验收等待可用的 Docker 环境

## 记录

### 2026-07-31

- [x] 盘点主线已有 Dockerfile、Compose、CLI 入口和数据目录配置。
- [x] 确认 API/Web UI 复用单一镜像，通过 `ka serve` / `ka webui` 区分模式。
- [x] 选择轻量构建参数方案：默认 `loaders` extra，OCR 系统包默认不安装。
- [x] 修复项目本身未安装导致 `ka` CLI 入口可能缺失的问题。
- [x] 增加本地 embedding/PyTorch 所需的 `libgomp1` 运行时依赖。
- [x] 将运行阶段切换为固定 UID/GID `10001` 的非 root 用户。
- [x] 为 API/Web UI 增加独立健康检查和可配置宿主机端口。
- [x] 移除 Web UI 对 API 的伪依赖，允许两个服务独立启动。
- [x] 使用具名卷持久化 `/app/data`，并支持自定义卷名。
- [x] 增加 `.env.example` 以及构建上下文的密钥、数据和缓存排除规则。
- [x] 使用结构化 YAML 解析器确认 API/Web UI 定义、健康检查和数据卷继承正确。
- [ ] 执行 `docker compose config`（当前环境未安装 Docker CLI）。
- [ ] 构建默认镜像并验证 CLI、用户身份和 API/Web UI 健康状态。
- [ ] 验证 OCR 构建参数和容器重建后的数据持久性。

## 当前限制

- 当前开发环境没有 `docker`、`docker-compose`、`podman` 或 `nerdctl`，因此本轮不执行
  镜像构建和容器运行验收。
- 按本阶段 spec 保留了完整验收命令，后续可在具备 Docker Compose 的环境继续记录结果。
