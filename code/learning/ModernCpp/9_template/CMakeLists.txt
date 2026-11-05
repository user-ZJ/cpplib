cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)

project(MyProject)

# 查找当前目录下所有的.cpp文件，包括子文件夹中的文件，并将名称保存到变量中
file(GLOB_RECURSE SOURCES "*.cpp")

# 从源文件列表中删除名为"exclude.cpp"的文件
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/3_using_alias.cpp")

# 循环处理所有源文件
foreach(sourcefile ${SOURCES})
    # 获取源文件的名称
    get_filename_component(name ${sourcefile} NAME_WE)

    # 指定生成目标
    add_executable(${name} ${sourcefile})
endforeach()
