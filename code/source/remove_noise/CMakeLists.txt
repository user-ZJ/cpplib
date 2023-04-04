cmake_minimum_required(VERSION 3.14)

include_directories(${CMAKE_CURRENT_LIST_DIR})

link_directories()

aux_source_directory(. SRC)

add_library(remove_noise SHARED ${SRC})
target_link_libraries(remove_noise rnnoise glog gflags sox)
