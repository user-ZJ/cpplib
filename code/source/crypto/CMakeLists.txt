cmake_minimum_required(VERSION 3.14)
project(crypto)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )

if(NOT DEFINED crypto_wrapper_SOURCE_DIR)
set(crypto_wrapper_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
aux_source_directory(. SOURCE_LIST)

add_library(crypto_wrapper STATIC ${SOURCE_LIST})
target_link_libraries(crypto_wrapper crypto ssl myutil)
endif()
