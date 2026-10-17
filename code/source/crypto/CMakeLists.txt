cmake_minimum_required(VERSION 3.14)
project(crypto)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )

if(NOT DEFINED crypto_wrapper_SOURCE_DIR)
set(crypto_wrapper_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
# aux_source_directory(. SOURCE_LIST)
set(SOURCE_LIST AESCryptoWrapper.cpp)

find_package(OpenSSL REQUIRED)
add_library(crypto_wrapper STATIC ${SOURCE_LIST})
# 只用到 utils 中的头文件，不链接 myutil，避免引入 sox 等无关依赖
target_link_libraries(crypto_wrapper PUBLIC OpenSSL::Crypto glog)
endif()
