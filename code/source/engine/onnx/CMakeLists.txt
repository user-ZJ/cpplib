project(onnx_engine)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
	${CMAKE_CURRENT_LIST_DIR}
 )


#  add_definitions(-DDEBUG)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/acoustic ACOUS_ONNX_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/vocoder VOCODER_ONNX_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/g2p G2P_ONNX_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/prosody PROSODY_ONNX_SRC)



add_library(tts_engine OBJECT ${ACOUS_ONNX_SRC} ${VOCODER_ONNX_SRC} ${G2P_ONNX_SRC} ${PROSODY_ONNX_SRC}) 

target_link_libraries(tts_engine onnxruntime)

