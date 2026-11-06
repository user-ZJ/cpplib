# mbase（cpplib）

## 问题与范围

本仓库是一套 **C++17 基础库与示例程序**，覆盖：

- HTTP / WebSocket（Poco）
- 加解密（OpenSSL AES；Poco RSA 实现未编入默认库）
- 音频工具、降噪（sox / sonic / rnnoise）
- 推理相关目录（ONNX Runtime、可选 TensorRT / libtorch / MNN 等）
- 可选 Kaldi / OpenFST（MKL 或 OpenBLAS 后端）
- 可选 MySQL / Redis / MongoDB 封装（依赖对应客户端）
- gRPC 示例（Linux 预编译包或本机安装）

**不是** 面向终端用户的完整产品；主要供内部服务与算法工程复用模块、命令行工具与测试。

## 受众

- 使用本库的 C++ 服务 / 算法开发者
- 在 Linux（主目标）或 macOS（开发机）上构建与调试的工程师
- 通过 Cursor / Claude Code 等在本仓库协作的 AI agent

## 当前边界

- 构建系统：CMake ≥ 3.14，C++17，命名空间宏 `BASE_NAMESPACE`（默认 `zack`）
- 依赖通过 `cmake/*.cmake` 查找本地安装；安装命令和脚本见 README，CMake 不下载依赖
- macOS 使用 OpenBLAS 替代 MKL；未安装 BLAS 时跳过 Kaldi，VAD 还需 fvad/ONNX Runtime；无 MySQL 客户端时跳过 `source/db`
- UI / 前端：无产品级界面；`source/web` 为网络模块，服务入口位于 `apps/web`

## MVP / 验收（库级）

1. 在支持平台上 `cmake .. && make` 能编出已启用的目标。
2. 与变更相关的 `tests/`（GoogleTest，`ctest -R unit_test`）或 `examples/` 示例程序能跑通。
3. Agent 与开发者能通过 `AGENTS.md`、`docs/` 找到构建命令、模块边界与已知限制。

## 假设与未决

- 生产部署、发布流水线、统一版本号策略未在本仓库固化。
- `3rdparty/readme.md` 中部分版本号可能落后于 `cmake/*.cmake` 实际拉取版本，以 CMake 脚本为准。
- 原根目录文件 `docs`（FastAPI Swagger 快照）已改名为 `docs/archive/fastapi-swagger.html`，与正式文档目录 `docs/` 无关。
