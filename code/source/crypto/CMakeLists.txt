cmake_minimum_required(VERSION 3.14)
project(crypto)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )


aux_source_directory(. SOURCE_LIST)

add_library(crypto_wrapper STATIC ${SOURCE_LIST})
target_link_libraries(crypto_wrapper PocoCrypto PocoFoundation crypto ssl utils)

