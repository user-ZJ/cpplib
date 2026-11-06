cmake_minimum_required(VERSION 3.14)
project(vadwrapper)



include_directories(
		 )
link_directories(
)


add_library(vadwrapper FVadWrapper.cpp SileroVadWrapper.cpp aliVadWrapper.cpp)
target_link_libraries(vadwrapper PUBLIC fvad onnxruntime kaldi-feat myutil)


include(fvad)
target_include_directories(vadwrapper PUBLIC "${CMAKE_SOURCE_DIR}/source/kaldi")
target_link_libraries(vadwrapper PUBLIC kaldi_blas)
