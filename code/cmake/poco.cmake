# 不强制 ENABLE_DATA_MYSQL，由 Poco 自动探测：找到 MySQL 客户端才构建 Poco::DataMySQL
# 依赖它的 source/db 及相关 test 会随之跳过。需要时安装：
#   Ubuntu: apt install libmysqlclient-dev
#   macOS:  brew install mysql-client，并传 -DMYSQL_ROOT_DIR=$(brew --prefix mysql-client)
FetchContent_Declare(poco
  URL      https://github.com/pocoproject/poco/archive/refs/tags/poco-1.14.2-release.tar.gz
  URL_HASH SHA256=47394ea7ddb7b0a40e1a5be896f8f5dc77cfdc4f561d2e7131ecf582df5a0c3a
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