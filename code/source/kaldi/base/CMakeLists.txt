cmake_minimum_required(VERSION 3.14)
project(kaldi-base)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
)

link_directories(
    /opt/intel/mkl/lib/intel64
    /opt/intel/lib/intel64/
)

aux_source_directory(. SOURCE_LIST)

set(MKL_LIBS mkl_intel_lp64 mkl_core mkl_sequential)

add_library(${PROJECT_NAME} STATIC ${SOURCE_LIST})
target_link_libraries(${PROJECT_NAME} fst ${MKL_LIBS})

add_dependencies(${PROJECT_NAME} openfst)