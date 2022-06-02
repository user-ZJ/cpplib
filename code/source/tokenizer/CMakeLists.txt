cmake_minimum_required(VERSION 3.5)
project(tokenizer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
		 )

link_directories(
        )

set(HEADERS Tokenizer.h)

set(SOURCES_LIST
		Tokenizer.cpp
		)

add_library(${PROJECT_NAME} STATIC ${SOURCES_LIST})

target_link_libraries(${PROJECT_NAME} glog gflags)



