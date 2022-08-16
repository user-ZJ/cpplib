
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
	${CMAKE_CURRENT_LIST_DIR}
 )



aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/g2p G2P_ENGINE_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/prosody PROSODY_ENGINE_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/punct PUNCT_ENGINE_SRC)



add_library(text_process_engine STATIC ${G2P_ENGINE_SRC} ${PROSODY_ENGINE_SRC} ${PUNCT_ENGINE_SRC})

target_link_libraries(text_process_engine MNN glog gflags)

