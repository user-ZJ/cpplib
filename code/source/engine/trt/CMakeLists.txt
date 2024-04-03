project(onnx_engine)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
	${CMAKE_CURRENT_LIST_DIR}
 )


#  add_definitions(-DDEBUG)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/embedding EMBEDDING_ONNX_SRC)


find_package(CUDA REQUIRED)

add_library(embedding_engine STATIC ${EMBEDDING_ONNX_SRC}) 

target_link_libraries(embedding_engine nvinfer nvinfer_plugin nvparsers nvonnxparser cudart crypto_wrapper)

