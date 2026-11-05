# 架构

## 总览

工程名 **mbase**（CMake project）。公共 C++ 命名空间由宏 `BASE_NAMESPACE` 控制（默认 `zack`）。源码按模块放在 `source/<module>/`，根 `CMakeLists.txt` 用 `add_subdirectory` 按条件加入。

```
应用 / 测试可执行文件
        │
   source/* 静态或共享库
        │
 cmake/*.cmake → 本地 find_package / find_path / find_library → 本地安装前缀
```

## 模块边界（`source/`）

| 模块 | 职责（概括） | 典型产物 |
| --- | --- | --- |
| `web` | HTTP/WebSocket（Poco）实现与头文件 | 由 `apps/web` 编入 `web_server_bin` |
| `crypto` | AES 封装 | `crypto_wrapper` |
| `utils` | 文件、字符串、线程、计时、音频与 Tensor 等工具 | `myutil` |
| `audio` / `remove_noise` | 音频与降噪 | `remove_noise` 等 |
| `engine` / `onnx` / … | 推理相关 | 条件开启 |
| `kaldi` / `openfst` / `vad` | 语音相关 | 常需 MKL；macOS 可跳过 |
| `db` | MySQL 等 | 无客户端则跳过 |

具体 `target_link_libraries` 以各模块 `CMakeLists.txt` 为准。

## 依赖管理

- **入口**：`cmake/` 下按库拆分的 `.cmake` 文件，由根或模块 `include`。
- **安装**：默认 `.local-deps/`，由安装脚本显式生成；`fc_base/` 仅保留为历史缓存。
- **本机**：Homebrew / 系统 OpenSSL、可选 MySQL、MKL 等；缺依赖时用 `SKIP_*` 或 CMake 探测跳过，而不是硬失败（视脚本而定）。

## 条件编译要点

- **MKL**：影响 OpenFST / Kaldi / 部分 VAD 路径。
- **MySQL**：影响 `source/db`。
- **gRPC**：所有平台使用本地安装，示例位于 `examples/grpc/`。
- **macOS**：开发友好路径上常关闭 MKL 相关子树；文件系统与 OpenSSL 路径需与 SDK 一致。

## 非目标

- 不提供统一微服务编排或产品级前端。
- 不把 `docs/archive/fastapi-swagger.html` 当作本库 API 契约（历史快照）。

## 目录职责

- `source/`：可复用库；本轮保持公共头文件路径和库目标不变。
- `apps/`：程序入口；`crypto/` 为 AES 工具，`web/` 为服务端。
- `examples/`：模块、网络客户端及 gRPC 示例，沿用 `BUILD_TEST` 开关和原可执行目标名。
- `tests/`：GoogleTest 自动化测试。
- `learning/`：ModernCpp、MPI 学习工程，各自独立构建。
- `experiments/`：CUDA、ONNX、TensorRT 实验工程，不加入根工程默认构建；需各自的 SDK 和平台环境。
- `3rdparty/`：依赖说明及已有预编译包；`3rdparty/patches/`：依赖补丁；`packages/`：已有离线源码包。
- `fc_base/`、`build/`：本地依赖缓存及构建产物，不提交。

`source/engine`、`source/tokenizer` 尚未加入根工程；`source/audio`、`source/params` 也不作为独立子目录构建。根 CMake 的实际目标列表为构建范围依据。

目录迁移：`test/` → `examples/`，`ModernCpp/`、`MPI/` → `learning/`，`cudacode/`、`onnx/`、`trt/` → `experiments/`。旧路径的外部脚本需相应更新。

## 构建路径作用范围

自有目标通过 target 级 PRIVATE include 保留现有 `source/` 头文件兼容入口；`myutil` 和 `crypto_wrapper` 同时通过 PUBLIC include 声明导出该路径。MKL/Kaldi 路径仅作用于 Kaldi 子目录；本地 opus/fvad imported target 向包装目标传递 SDK 路径。示例不再全局引入 CUDA 路径。后续新增依赖应优先使用 imported target 和 target 级设置。

显式安装阶段的 Poco 补丁接受原始源码及已应用状态；不匹配时配置明确失败，升级依赖时必须审核补丁。

项目编译策略集中在 `cmake/project_targets.cmake`，仅应用于根工程已启用的自有目标：要求 C++17、PIC、Debug 的 DEBUG 宏以及 GNU/Clang 的快速错误停止与非 Debug 优化。Debug 符号由 CMake 标准配置提供；第三方目标不再继承根工程追加的这些参数。用户仍可通过标准 CMake 编译参数配置工具链。

`tests/CMakeLists.txt` 显式列出测试源文件，新增测试时同步更新该列表，使构建系统明确跟踪源文件集合变化。

## 应用与测试目录迁移

- `source/bin/` → `apps/crypto/`。
- `source/web/WebServerBin.cpp` → `apps/web/`，Web 模块的实现与头文件继续位于 `source/web/`。
- `source/web/HTTPClient.cpp`、`WebSocketClient.cpp` → `examples/network/`。
- `source/grpc/` → `examples/grpc/`，协议在其 `proto/` 下，生成文件只写入构建目录。
- `unittest/` → `tests/`，保留 `unit_test` 目标及 CTest 名称。
- `patch/` → `3rdparty/patches/`，安装脚本已同步；历史 Swagger 页面归入 `docs/archive/`。

库代码继续采用 `source/<module>` 布局，不强行迁移到 `src/include`，避免破坏既有公共头文件包含路径。`BUILD_TEST=OFF` 关闭全部示例（包括网络客户端和 gRPC），应用仍构建。
