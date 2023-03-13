cmake_minimum_required(VERSION 3.14)
project(utils)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
# link_directories(
# 	/home/dmai/code/poco-1.11.1-all/build/lib
# 		)


add_library(${PROJECT_NAME} STATIC audio-util.cpp)
target_link_libraries(${PROJECT_NAME} glog gflags_nothreads sox sonic pthread)
#target_link_libraries(utils glog gflags sox pthread)


