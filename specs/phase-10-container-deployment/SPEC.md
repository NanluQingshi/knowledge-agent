# Phase 10 Spec: Docker 与本地部署

## 目标

为 API 和 Web UI 提供可重复的一键部署方式，使新环境不需要手动安装 Python
和系统依赖，并确保运行时密钥、本地数据与镜像构建过程隔离。

## 当前基线

- 主线已有多阶段 `Dockerfile` 和 API/Web UI 两个 Compose 服务。
- 现有镜像默认安装全部可选依赖和 OCR 系统包，体积与构建时间不可控。
- 现有运行阶段使用 root，且没有服务级健康检查。
- 现有构建使用 `poetry install --no-root`，没有保证 `ka` CLI 入口被安装。
- 现有 `.dockerignore` 没有排除 `.env`，存在把本地密钥复制进镜像的风险。

## 实现范围

### 镜像

- 保留单一多阶段镜像，API 和 Web UI 通过不同命令启动。
- 显式安装项目本身，确保 `ka` CLI 入口存在。
- 运行镜像包含本地 embedding/PyTorch 所需的 `libgomp1` 动态库。
- 运行阶段固定使用 UID/GID `10001` 的非 root 用户。
- 默认安装 `loaders` extra；通过 `POETRY_EXTRAS` 构建参数选择其他 extra。
- OCR 系统包通过 `INSTALL_OCR=true` 构建参数按需安装，默认关闭。

### Compose

- API 和 Web UI 复用同一镜像定义。
- API 使用 `/health`，Web UI 使用根路径执行独立健康检查。
- 使用具名卷挂载 `/app/data`，容器重建后数据仍保留。
- 支持通过环境变量覆盖端口、卷名、镜像标签和构建选项。
- API 和 Web UI 没有伪造的服务依赖，可单独启动任一模式。

### 配置与安全

- 提供不含真实密钥的 `.env.example`。
- `.gitignore` 和 `.dockerignore` 排除 `.env` 及其变体。
- 镜像中不复制测试、数据、VCS 元数据和开发缓存。

## 非目标

- 不发布远端镜像或配置镜像仓库。
- 不引入 Kubernetes、反向代理、TLS 或云平台清单。
- 不加入 GPU/CUDA 镜像变体。
- 不在本阶段改造应用层存储协议或工作流逻辑。

## 验收标准

- `docker compose config` 可解析完整配置。
- 默认镜像可构建，容器内 `id -u` 返回非 `0`，且 `ka --version` 可执行。
- `KA_DOCKER_INSTALL_OCR=true` 时镜像中可执行 `tesseract --version`。
- `docker compose up -d api` 后 API 健康状态为 `healthy`，宿主机 `/health` 可访问。
- `docker compose up -d webui` 后 Web UI 健康状态为 `healthy`，宿主机首页可访问。
- 重建服务后具名卷中的知识库数据仍存在。
- 镜像文件系统和历史中不包含本地 `.env`、`data/` 或开发缓存。

## 前置条件

Phase 9 CI 和入口点修复完成。

## 验证命令

```bash
docker compose config
docker compose build
docker compose up -d api
docker compose ps
curl --fail http://localhost:${KA_API_PORT:-8000}/health
docker compose run --rm api id -u
docker compose run --rm api ka --version
```
