project(onnx_engine)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
	${CMAKE_CURRENT_LIST_DIR}
 )


#  add_definitions(-DDEBUG)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/wav2vec WAV2VEC_ONNX_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/acoustic ACOUSTIC_ONNX_SRC)




add_library(bs_onnx_engine STATIC ${WAV2VEC_ONNX_SRC} ${ACOUSTIC_ONNX_SRC}) 

target_link_libraries(bs_onnx_engine onnxruntime crypto_wrapper glog gflags_nothreads)

