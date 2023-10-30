cmake_minimum_required(VERSION 3.13)
project(MyMPIProject)

# 设置用于找到MPI的路径
set(MPI_DIR "/usr/local/mpi")

# 查找MPI
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# 添加头文件路径
include_directories(
    /usr/local/cuda/include
    ${MPI_INCLUDE_PATH}
)

# 设置源文件列表
set(SOURCES main.cpp)

# 添加可执行文件，并链接MPI库
add_executable(my_mpi_program ${SOURCES})
target_link_libraries(my_mpi_program ${MPI_CXX_LIBRARIES})


cuda_add_executable(test test.cu)
target_link_libraries(test ${CUDA_LIBRARIES} ${MPI_CXX_LIBRARIES})