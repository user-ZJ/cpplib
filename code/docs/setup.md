# 环境搭建

## 前置

- CMake ≥ 3.14
- 支持 C++17 的编译器（GCC / Clang / Apple Clang）
- 网络：首次配置需拉取 FetchContent / ExternalProject 依赖到 `fc_base/`
- macOS：建议 Xcode CLT；OpenSSL 等可用 Homebrew

## 构建

在 `code/` 目录：

```bash
mkdir -p build && cd build
cmake ..
make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
```

单目标示例：

```bash
make web_server_bin
make crypto_wrapper
make remove_noise
make unit_test
ctest --output-on-failure -R unit_test
```

`unit_test` 覆盖 utils（含文件、计时、UUID、线程池、FFT、NDTensor）、路径目录、不依赖服务的 HTTP 工具函数，以及 AES。关闭单元测试：`cmake -DBUILD_UNITTEST=OFF ..`。

## 可选依赖

| 能力 | 典型需求 | 缺失时 |
| --- | --- | --- |
| Kaldi / OpenFST / 部分 VAD | MKL 等 | 跳过相关 `add_subdirectory` |
| `source/db` | MySQL 客户端 | 跳过 db |
| gRPC 示例 | 预编译包或本机 gRPC | 按平台 CMake 逻辑处理 |

具体开关与探测逻辑见根 `CMakeLists.txt` 与 `cmake/`。

## 配置与数据

- 运行参数、证书、模型路径以各模块 README / 示例代码为准。
- 勿把真实密钥写入仓库；本地覆盖用未跟踪配置文件。

## 故障排查

- 优先看 `README.md` 中已记录的 macOS / OpenSSL / filesystem / OpenFST / rnnoise 等问题。
- `3rdparty/readme.md` 版本可能过时，以 `cmake/*.cmake` 为准。
- 清理：删除 `build/` 与必要时部分 `fc_base/` 后重新 `cmake`。
