cmake_minimum_required(VERSION 3.14)
project(kaldiasr)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DKALDI_DOUBLEPRECISION=0 -DHAVE_CUDA=0)
# Directory scope is confined to the legacy Kaldi subtree.
link_libraries(kaldi_blas)

add_subdirectory(base)
add_subdirectory(matrix)
add_subdirectory(cudamatrix)
add_subdirectory(util)
add_subdirectory(tree)
add_subdirectory(gmm)
add_subdirectory(hmm)
add_subdirectory(lat)
add_subdirectory(fstext)
add_subdirectory(transform)
add_subdirectory(feat)
add_subdirectory(bin)
add_subdirectory(decoder)
add_subdirectory(ivector)
add_subdirectory(chain)
add_subdirectory(nnet2)
add_subdirectory(nnet3)
add_subdirectory(online2)



