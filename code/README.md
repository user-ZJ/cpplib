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

# 编译

## linux

```shell
cd code/root/path
mkdir build
cd build
cmake .. #如果需要编译benchmark，则使用cmake -DBUILD_BENCHMARK=ON ..
# cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j 4
```

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

