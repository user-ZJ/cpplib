cmake_minimum_required(VERSION 3.14)
project(kaldiasr)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


add_definitions(-DHAVE_MKL)
add_definitions(-DKALDI_DOUBLEPRECISION=0)
add_definitions(-DHAVE_CUDA=0)

include_directories(
		${CMAKE_SOURCE_DIR}/kaldi
        ${CMAKE_SOURCE_DIR}/tools/include/openfst
		/opt/intel/mkl/include
        ${CMAKE_SOURCE_DIR}/tools/include/cub
        # /usr/local/cuda/include
		 )

link_directories(
	/usr/local/lib
    /opt/intel/mkl/lib/intel64
	/opt/intel/lib/intel64/
    # /usr/local/cuda/lib64
    )


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



