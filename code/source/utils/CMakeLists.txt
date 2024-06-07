cmake_minimum_required(VERSION 3.14)
project(myutil)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
# link_directories(
# 	/home/dmai/code/poco-1.11.1-all/build/lib
# 		)

aux_source_directory(. CPP_LIST)

add_library(${PROJECT_NAME} SHARED ${CPP_LIST})
target_link_libraries(${PROJECT_NAME} PUBLIC glog gflags::gflags sox sonic pthread)



