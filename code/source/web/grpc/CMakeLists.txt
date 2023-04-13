cmake_minimum_required(VERSION 3.5)

project(MyProject)

set(CMAKE_CXX_STANDARD 11)


set(CMAKE_PREFIX_PATH /data/code/grpc/cmake/install)


set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")


find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)


set(GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
# set(GRPC_CPP_PLUGIN_EXECUTABLE /data/code/grpc/cmake/install/bin/grpc_cpp_plugin)

message(STATUS "${PROTOBUF_LIBRARIES}")
message(STATUS "${PROTOBUF_INCLUDE_DIRS}")

set(GENS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gens)
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_LIST calculator helloworld)


foreach(_proto_file ${PROTO_LIST})
    get_filename_component(_proto_name ${_proto_file} NAME_WE)
    set(_proto_cpp_file "${GENS_DIR}/${_proto_name}.pb.cc")
    set(_proto_h_file "${GENS_DIR}/${_proto_name}.pb.h")
    set(_grpc_cpp_file "${GENS_DIR}/${_proto_name}.grpc.pb.cc")
    set(_grpc_h_file "${GENS_DIR}/${_proto_name}.grpc.pb.h")
    set(_grpc_file "${PROTO_DIR}/${_proto_file}.proto")
    add_custom_command(
        OUTPUT ${_proto_cpp_file} ${_proto_h_file} ${_grpc_cpp_file} ${_grpc_h_file}
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
        ARGS --grpc_out ${GENS_DIR}
        --cpp_out ${GENS_DIR}
        -I ${PROTO_DIR}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
        ${_grpc_file}
        DEPENDS ${_grpc_file} ${PROTOBUF_PROTOC_EXECUTABLE} ${GRPC_CPP_PLUGIN_EXECUTABLE} 
        COMMENT "Running grpc_cpp_plugin on ${_proto_file}"
        VERBATIM)
    list(APPEND GENS_SRCS ${_grpc_cpp_file})
    list(APPEND GENS_HDRS ${_grpc_h_file})
    list(APPEND GENS_SRCS ${_proto_cpp_file})
    list(APPEND GENS_HDRS ${_proto_h_file})
endforeach()

include_directories(${GENS_DIR})

add_executable(calculator_server calculator_server.cpp ${GENS_SRCS})
target_link_libraries(calculator_server gRPC::grpc++ gRPC::grpc++_reflection ${PROTOBUF_LIBRARIES})
# target_link_libraries(calculator_server gRPC::grpc++ gRPC::grpc++_reflection protobuf)