# 不强制 ENABLE_DATA_MYSQL，由 Poco 自动探测：找到 MySQL 客户端才构建 Poco::DataMySQL
# 依赖它的 source/db 及相关 test 会随之跳过。需要时安装：
#   Ubuntu: apt install libmysqlclient-dev
#   macOS:  brew install mysql-client，并传 -DMYSQL_ROOT_DIR=$(brew --prefix mysql-client)
FetchContent_Declare(poco
  URL      https://github.com/pocoproject/poco/archive/refs/tags/poco-1.13.0-release.tar.gz
  URL_HASH SHA256=0135160663795901f317215272fadf71f3b526f38daacb2ae8d6b07ad11d319b
  PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/patch/poco/fix_zutil_macos.cmake
)
FetchContent_MakeAvailable(poco)
include_directories(${poco_SOURCE_DIR}/Foundation/include 
                    ${poco_SOURCE_DIR}/Util/include
                    ${poco_SOURCE_DIR}/ActiveRecord/include
                    ${poco_SOURCE_DIR}/Crypto/include
                    ${poco_SOURCE_DIR}/Data/include
                    ${poco_SOURCE_DIR}/Data/MySQL/include
                    ${poco_SOURCE_DIR}/JSON/include
                    ${poco_SOURCE_DIR}/MongoDB/include
                    ${poco_SOURCE_DIR}/Encodings/include
                    ${poco_SOURCE_DIR}/Net/include 
                    ${poco_SOURCE_DIR}/NetSSL_OpenSSL/include
                    ${poco_SOURCE_DIR}/NetSSL_Win/include 
                    ${poco_SOURCE_DIR}/Redis/include
                    ${poco_SOURCE_DIR}/XML/include
                    ${poco_SOURCE_DIR}/Zip/include)
link_directories(${CMAKE_BINARY_DIR}/lib)