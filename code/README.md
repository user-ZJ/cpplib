# mbase / cpplib

C++17 基础库与示例（CMake）。产品范围见 [PRODUCT.md](./PRODUCT.md)；给 AI agent 的操作约定见 [AGENTS.md](./AGENTS.md)。

| 文档 | 说明 |
| --- | --- |
| [PRODUCT.md](./PRODUCT.md) | 问题、范围、验收 |
| [AGENTS.md](./AGENTS.md) | Agent 入口与变更约定 |
| [docs/architecture.md](./docs/architecture.md) | 模块与依赖 |
| [docs/quality.md](./docs/quality.md) | 构建 / 测试门禁 |
| [docs/setup.md](./docs/setup.md) | 环境搭建 |
| [docs/onboarding.md](./docs/onboarding.md) | 上手路径 |
| [docs/design.md](./docs/design.md) | UI（本库几乎无产品界面） |

## 本地依赖安装

CMake 仅查找本地安装，不下载、解压或编译第三方依赖。缺少必需依赖时配置失败；gRPC/Protobuf 和 MySQL 缺失时跳过对应模块。

macOS：

```bash
brew install cmake pkg-config openssl@3 sox opus grpc protobuf openblas
# 同时安装缺少的 sonic（源码构建）：
bash scripts/install_system_deps.sh
```

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config curl unzip git libssl-dev libsox-dev libsonic-dev libopenblas-dev liblapacke-dev libopus-dev libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc
# 或运行：bash scripts/install_system_deps.sh
```

安装项目所需固定版本依赖（默认安装到 `.local-deps/`，无需 sudo）：

```bash
INSTALL_ONNX=1 bash scripts/install_deps.sh
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j 4
ctest --test-dir build-local --output-on-failure -R unit_test
```

脚本安装 gflags 2.3.1、glog 0.7.1、Poco 1.14.2（含本地 OpenSSL 兼容补丁）、GoogleTest 1.11.0、旧版 rnnoise、Boost 1.87.0、RapidJSON 1.1.0、Eigen 3.4.1 和定制 cereal v1.3.2-beta.5。下载归档校验 SHA256；安装工作目录为 `.deps-work/`。`INSTALL_ONNX=1` 额外安装 ONNX Runtime 1.12.0；不需要推理时可省略并配置 `-DUSE_ONNXRUNTIME=OFF`。

```bash
# 自定义安装位置，配置时使用同一路径：
MBASE_DEPS_PREFIX=/path/to/deps bash scripts/install_deps.sh
cmake -S . -B build-local -DMBASE_DEPS_PREFIX=/path/to/deps -DUSE_ONNXRUNTIME=OFF
# 复用历史下载源码（可信缓存，不重新校验内容；仅供迁移）：
MBASE_SOURCE_CACHE="$PWD/fc_base" INSTALL_ONNX=1 bash scripts/install_deps.sh
```

也可自行安装，通过 `CMAKE_PREFIX_PATH` 或各包的 `<Package>_DIR` 指定位置。cereal 必须提供 `cereal/archives/optional_json.hpp`，普通上游发行版不能替代该定制分支。安装脚本不会被 CMake 自动调用。

Kaldi 可使用 MKL 或 OpenBLAS；显式安装脚本默认安装 OpenFST 1.6.5（`INSTALL_KALDI=0` 可跳过）（含已有兼容补丁）。VAD 额外需要 fvad 和 ONNX Runtime，脚本不安装 fvad。MySQL 可选：macOS `brew install mysql-client`，Ubuntu `sudo apt-get install libmysqlclient-dev`，同时需本地 Poco 构建提供 DataMySQL 组件（脚本默认关闭）。Android 必须预先安装面向目标 ABI 的依赖，不能使用宿主机库。

## 编译

使用新的 `build-local/` 目录，避免旧 `build/` 中 FetchContent 的缓存路径残留。

## Android

```shell
# 编译64位库
export ANDROID_NDK=/opt/env/android-ndk-r22b
mkdir build_android64
cd build_android64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_NDK=$ANDROID_NDK -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_static ..
make -j 4

# 编译32位库
export ANDROID_NDK=/opt/env/android-ndk-r22b
mkdir build_android32
cd build_android32
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_NDK=$ANDROID_NDK -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_static ..
make -j 4
```


## 目录导航

```text
code/
├── source/       # 库实现与现有公共头文件
├── apps/         # AES 命令行工具、Web 服务入口
├── examples/     # 模块、网络客户端、gRPC 演示
├── tests/        # GoogleTest 单元测试
├── cmake/        # 本地依赖查找与项目编译策略
├── scripts/      # 显式安装依赖的脚本
├── 3rdparty/     # 依赖说明、patches/ 和历史资产
├── docs/         # 文档与 archive/ 历史文档
├── data/         # 模型与示例数据
├── dockerfiles/  # 容器配置
├── learning/     # 独立学习工程
└── experiments/  # 需额外 SDK 的独立实验
```

`apps/` 始终构建；`examples/` 由 `BUILD_TEST` 控制，包含网络客户端和 gRPC 演示（缺少 SDK 时跳过 gRPC）。
- `tests/`：GoogleTest（`BUILD_UNITTEST`）。
- `learning/ModernCpp`、`learning/MPI`：独立学习工程。
- `experiments/cudacode`、`experiments/onnx`、`experiments/trt`：需要额外 SDK 的独立实验工程。
- 构建产物在 `build/`，本地依赖安装在 `.local-deps/`（`fc_base/` 仅为历史缓存）。详细迁移说明见 `docs/architecture.md`。

验证默认工程：

```bash
cmake -S . -B build-local
cmake --build build-local -j 8
ctest --test-dir build-local --output-on-failure -R unit_test
```

## macOS 使用 OpenBLAS

Kaldi 在 macOS 的 AUTO 模式使用 OpenBLAS，其他平台默认 MKL。依赖均须预先本地安装；Kaldi 还需兼容本仓库旧代码的 OpenFST，VAD 额外需要 fvad 和 ONNX Runtime。

```bash
brew install openblas
INSTALL_KALDI=1 bash scripts/install_deps.sh
cmake -S . -B build-local -DKALDI_BLAS_BACKEND=OPENBLAS
cmake --build build-local --target kaldi-matrix kaldi-feat -j 4
```

可选值为 `AUTO`、`OPENBLAS`、`MKL`。显式指定后端时缺依赖会报错，AUTO 未找到后端时跳过 Kaldi。可用 `-DBUILD_KALDI=OFF` 关闭整个 Kaldi/VAD 子树。自定义位置用 `CMAKE_PREFIX_PATH` 或 `OPENBLAS_INCLUDE_DIR`、`OPENBLAS_LAPACKE_INCLUDE_DIR`、`OPENBLAS_LIBRARY` 指定。使用 LP64（32 位 BLAS 整数）版本，不能替换为 ILP64 库。
