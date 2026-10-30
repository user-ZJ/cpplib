# Agent 指南 — mbase / cpplib

本文件是 AI agent 与自动化工具的入口。人类可读产品说明见 [PRODUCT.md](./PRODUCT.md)；详细文档见 [docs/](./docs/)。

## 技术栈（固定）

| 项 | 约定 |
| --- | --- |
| 语言 | C++17 |
| 构建 | CMake ≥ 3.14 + Make/Ninja |
| 测试 | 各模块下可执行测试 / 示例（非统一 Node/Vitest） |
| 包管理 | 不用 npm/pnpm；第三方由 `cmake/*.cmake` + `fc_base/` |
| 前端 | 无 Next.js / ESLint；勿引入 Node 质量门禁 |

**禁止**为「对齐 vibe 模板」改成 TypeScript/Next.js/Drizzle，或改动现有 CMake 依赖图，除非用户明确要求。

## 目录速览

```
code/
├── CMakeLists.txt          # 根工程 mbase
├── cmake/                  # 依赖与选项（*.cmake）
├── source/                 # 业务与基础库模块
├── test/                   # 测试 / 示例源码（部分随模块目标生成）
├── 3rdparty/               # 说明与本地补丁；大依赖多在 fc_base
├── docs/                   # 架构 / 质量 / 搭建文档
├── PRODUCT.md
├── AGENTS.md               # 本文件
└── CLAUDE.md               # Claude Code：@AGENTS.md
```

主要模块（`source/<name>`）与目标名大致对应，例如：`web` → `web` / `web_server_bin`；`crypto` → `crypto` / `crypto_wrapper`；`remove_noise` → `remove_noise`。

## 文档地图

| 文档 | 用途 |
| --- | --- |
| [PRODUCT.md](./PRODUCT.md) | 问题、范围、受众、验收 |
| [docs/architecture.md](./docs/architecture.md) | 模块边界与依赖 |
| [docs/quality.md](./docs/quality.md) | 构建 / 测试 / 变更门禁 |
| [docs/setup.md](./docs/setup.md) | 环境与配置步骤 |
| [docs/onboarding.md](./docs/onboarding.md) | 新人 / 新 agent 上手 |
| [docs/design.md](./docs/design.md) | UI/设计说明（本库几乎无产品 UI） |
| [README.md](./README.md) | 人类向构建与问题排查 |

## 常用命令

在 `code/` 下（或仓库习惯的 out-of-source 构建目录）：

```bash
mkdir -p build && cd build
cmake ..
make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
```

按目标编译示例：

```bash
make web_server_bin
make crypto_wrapper
make remove_noise
```

选项（根 `CMakeLists.txt` / `cmake/`）：关注 `SKIP_*`、MKL、MySQL、gRPC 等条件编译；macOS 常跳过 MKL 相关与部分 DB。

**不要**假设存在 `npm test` / `eslint`。质量门禁见 [docs/quality.md](./docs/quality.md)。

## 变更约定

1. **先读再改**：动模块前看该目录 `CMakeLists.txt` 与同名头文件；改公共 API 时同步调用方与测试。
2. **小步提交**：一次只改一个关注点；不顺手大范围格式化无关文件。
3. **构建优先**：改 C++/CMake 后至少配置并编译受影响目标；能跑相关测试则跑。
4. **依赖**：新增第三方须走现有 `cmake/*.cmake` 模式；勿把二进制大文件提交进 git（见 `.gitignore` 与 `fc_base/`）。
5. **文档**：行为或构建方式变化时更新 `docs/` 与必要时 `README.md` / `PRODUCT.md`。
6. **密钥**：勿提交证书私钥、生产连接串；示例配置用占位符。

## 已知坑（摘要）

- 根目录曾有名为 `docs` 的 HTML 文件；现为 `docs-legacy-fastapi-swagger.html`。正式文档在 `docs/`。
- `filesystem` / OpenSSL / OpenFST / rnnoise 等在 macOS 上需与系统 SDK、Homebrew 对齐；细节见 README 与 architecture。
- `3rdparty/readme.md` 版本列表可能过时，以 `cmake/*.cmake` 为准。

## 完成定义（Definition of Done）

- [ ] 相关目标可配置、可编译
- [ ] 相关测试或手跑示例通过（若适用）
- [ ] 无无关大范围 diff
- [ ] 对外行为变化已反映在 docs / PRODUCT
- [ ] 未引入 Node/前端质量栈，未擅自改依赖策略
