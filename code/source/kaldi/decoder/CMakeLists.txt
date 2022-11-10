cmake_minimum_required(VERSION 3.14)
project(kaldi-decoder)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
)

aux_source_directory(. SOURCE_LIST)

add_library(${PROJECT_NAME} STATIC ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME} kaldi-gmm kaldi-hmm kaldi-lat kaldi-transform kaldi-fstext)


