project(tflite_engine)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
	${CMAKE_CURRENT_LIST_DIR}
 )


#  add_definitions(-DDEBUG)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/hvad hvad_TFLITE_SRC)


add_library(hvad_engine OBJECT ${hvad_TFLITE_SRC}) 

target_link_libraries(hvad_engine tensorflowlite glog gflags)

