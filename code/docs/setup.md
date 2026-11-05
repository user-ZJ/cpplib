# 环境搭建

安装与配置命令以 [README](../README.md) 为准。

先显式执行 `scripts/install_system_deps.sh` 安装系统开发依赖，再执行 `INSTALL_ONNX=1 scripts/install_deps.sh` 安装固定版本项目依赖，最后使用新的 `build-local/` 配置和编译。CMake 不执行网络下载，不编译第三方源码。

安装前缀由 `MBASE_DEPS_PREFIX` 指定（默认 `.local-deps/`），也支持 `CMAKE_PREFIX_PATH` 和包的配置目录。跨平台与交叉编译时依赖必须匹配目标架构。

Poco 的 OpenSSL 兼容补丁仅在显式安装脚本中应用，不修改系统安装。
