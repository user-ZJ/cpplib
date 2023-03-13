cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)


# 查找当前目录下所有的.cpp文件，包括子文件夹中的文件，并将名称保存到变量中
file(GLOB_RECURSE SOURCES "*.cpp")


# 指定生成目标
add_executable(pimlp_v1 ${SOURCES})
