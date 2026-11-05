cmake_minimum_required(VERSION 3.14)
project(myutil)



include_directories(
		 )
# link_directories(
# 	/home/dmai/code/poco-1.11.1-all/build/lib
# 		)

aux_source_directory(. CPP_LIST)

add_library(${PROJECT_NAME} SHARED ${CPP_LIST})
target_link_libraries(${PROJECT_NAME} PUBLIC glog gflags::gflags sox sonic pthread)




target_include_directories(myutil PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
