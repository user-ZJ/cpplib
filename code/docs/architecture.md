# 架构

## 总览

工程名 **mbase**（CMake project）。公共 C++ 命名空间由宏 `BASE_NAMESPACE` 控制（默认 `zack`）。源码按模块放在 `source/<module>/`，根 `CMakeLists.txt` 用 `add_subdirectory` 按条件加入。

```
应用 / 测试可执行文件
        │
   source/* 静态或共享库
        │
 cmake/*.cmake → FetchContent / ExternalProject → fc_base/
```

## 模块边界（`source/`）

| 模块 | 职责（概括） | 典型产物 |
| --- | --- | --- |
| `web` | HTTP/WebSocket（Poco） | `web`、`web_server_bin` |
| `crypto` | AES 等；wrapper 工具 | `crypto`、`crypto_wrapper` |
| `util` / `file` / `thread` / `timer` / … | 通用工具 | 对应库目标 |
| `audio` / `remove_noise` | 音频与降噪 | `remove_noise` 等 |
| `engine` / `onnx` / … | 推理相关 | 条件开启 |
| `kaldi` / `openfst` / `vad` | 语音相关 | 常需 MKL；macOS 可跳过 |
| `db` | MySQL 等 | 无客户端则跳过 |
| `grpc` | gRPC 示例 | 平台预编译或本机安装 |

具体 `target_link_libraries` 以各模块 `CMakeLists.txt` 为准。

## 依赖管理

- **入口**：`cmake/` 下按库拆分的 `.cmake` 文件，由根或模块 `include`。
- **缓存**：`fc_base/`（通常 gitignore），避免重复下载。
- **本机**：Homebrew / 系统 OpenSSL、可选 MySQL、MKL 等；缺依赖时用 `SKIP_*` 或 CMake 探测跳过，而不是硬失败（视脚本而定）。

## 条件编译要点

- **MKL**：影响 OpenFST / Kaldi / 部分 VAD 路径。
- **MySQL**：影响 `source/db`。
- **gRPC**：Linux 可能用预编译包；其它平台需本机安装。
- **macOS**：开发友好路径上常关闭 MKL 相关子树；文件系统与 OpenSSL 路径需与 SDK 一致。

## 非目标

- 不提供统一微服务编排或产品级前端。
- 不把 `docs-legacy-fastapi-swagger.html` 当作本库 API 契约（历史快照）。
