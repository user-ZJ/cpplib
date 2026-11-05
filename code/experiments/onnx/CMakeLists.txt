cmake_minimum_required(VERSION 3.14)
project(test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


include_directories(
	install/include
		 )
link_directories(
	install/lib
		)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

include(rapidjson)

set(CMAKE_PREFIX_PATH onnx/lib/cmake)

	
find_package(protobuf REQUIRED)
find_package(ONNX REQUIRED)

message("${ONNX_INCLUDE_DIRS}")
message("${protobuf_INCLUDE_DIRS}")
message("${Protobuf_VERSION}")

add_definitions(-DONNX_ML=1)

add_executable(test_onnx test_onnx.cpp)
target_link_libraries(test_onnx onnx_proto)




