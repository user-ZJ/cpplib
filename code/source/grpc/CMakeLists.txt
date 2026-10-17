cmake_minimum_required(VERSION 3.15)

project(MyProject)

# 3rdparty 中预编译的 grpc 1.55 只有 Ubuntu 20.04 x86_64 版本；
# 其他平台使用本机安装的 grpc，如 macOS: brew install grpc，或通过 -DCMAKE_PREFIX_PATH 指定。
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(GRPC_BUNDLED_PREFIX ${CMAKE_BINARY_DIR}/grpc)
    execute_process(COMMAND unzip -n -q ${CMAKE_SOURCE_DIR}/3rdparty/grpc_ubuntu20_install.zip -d ${GRPC_BUNDLED_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH ${GRPC_BUNDLED_PREFIX})
endif()

# protobuf >= 22 依赖 abseil，需用 CONFIG 模式才能拿到完整的链接依赖
find_package(Protobuf CONFIG QUIET)
find_package(gRPC CONFIG QUIET)
if(NOT Protobuf_FOUND OR NOT gRPC_FOUND)
    message(STATUS "gRPC/Protobuf not found, skip source/grpc (macOS: brew install grpc)")
    return()
endif()
message(STATUS "Using gRPC ${gRPC_VERSION}, Protobuf ${Protobuf_VERSION}")

set(PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
set(GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

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
        COMMAND ${PROTOC_EXECUTABLE}
        ARGS --grpc_out ${GENS_DIR}
        --cpp_out ${GENS_DIR}
        -I ${PROTO_DIR}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
        ${_grpc_file}
        DEPENDS ${_grpc_file} ${PROTOC_EXECUTABLE} ${GRPC_CPP_PLUGIN_EXECUTABLE}
        COMMENT "Running grpc_cpp_plugin on ${_proto_file}"
        VERBATIM)
    list(APPEND GENS_SRCS ${_grpc_cpp_file})
    list(APPEND GENS_HDRS ${_grpc_h_file})
    list(APPEND GENS_SRCS ${_proto_cpp_file})
    list(APPEND GENS_HDRS ${_proto_h_file})
endforeach()

include_directories(${GENS_DIR})

add_executable(calculator_server calculator_server.cpp ${GENS_SRCS})
target_link_libraries(calculator_server gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)

add_executable(calculator_client calculator_client.cpp ${GENS_SRCS})
target_link_libraries(calculator_client gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)
