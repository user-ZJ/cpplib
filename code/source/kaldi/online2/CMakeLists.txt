cmake_minimum_required(VERSION 3.14)
project(kaldi-online2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
)

aux_source_directory(. SOURCE_LIST)

add_library(${PROJECT_NAME} STATIC ${SOURCE_LIST})

target_link_libraries(${PROJECT_NAME} kaldi-chain kaldi-decoder kaldi-feat kaldi-gmm kaldi-hmm kaldi-ivector kaldi-lat kaldi-nnet2 kaldi-nnet3 kaldi-transform kaldi-tree)


