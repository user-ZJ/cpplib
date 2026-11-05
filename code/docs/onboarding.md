# 上手

## 30 分钟路径

1. 读 [PRODUCT.md](../PRODUCT.md) 弄清范围（库 + 工具，非完整 SaaS）。
2. 读 [architecture.md](./architecture.md) 选一个模块（建议从 `web` 或 `crypto` 开始）。
3. 按 [setup.md](./setup.md) 配置并编译对应目标。
4. 打开该模块 `source/<name>/` 与相关 `examples/`，跑一次示例二进制。
5. 改动前读 [AGENTS.md](../AGENTS.md) 与 [quality.md](./quality.md)。

## Agent 路径

1. 始终以 [AGENTS.md](../AGENTS.md) 为操作契约。
2. Claude Code：仓库根（本 `code/`）有 [CLAUDE.md](../CLAUDE.md) → `@AGENTS.md`。
3. 不改构建与依赖策略，除非用户明确要求。
4. 文档目录是 `docs/`；`docs/archive/fastapi-swagger.html` 可忽略。

## 协作习惯

- 问题优先用现有 CMake 选项表达，而不是复制一套新构建系统。
- PR / 变更说明写清：平台（Linux/macOS）、启用了哪些可选依赖、验证了哪些 target。
