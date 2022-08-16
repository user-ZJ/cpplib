set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
 )

if(USE_MNN)
	add_subdirectory(mnn)
elseif(USE_TFLITE)
	add_subdirectory(tflite)
else()
	add_subdirectory(onnx)
endif()