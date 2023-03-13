# apt install libmysqlclient-dev
set(ENABLE_DATA_MYSQL ON)
FetchContent_Declare(poco
  URL      https://pocoproject.org/releases/poco-1.12.4/poco-1.12.4-all.tar.gz
  URL_HASH SHA256=4c3584daa5b0e973f268654dbeb1171ec7621e358b2b64363cd1abd558a68777
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