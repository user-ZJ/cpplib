project(onnx_engine)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
	${CMAKE_CURRENT_LIST_DIR}
 )


#  add_definitions(-DDEBUG)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR} SOURCE_LIST)





add_library(bs_torch_engine STATIC ${SOURCE_LIST} ) 

target_link_libraries(bs_torch_engine torch torch_cpu crypto_wrapper glog gflags_nothreads)

