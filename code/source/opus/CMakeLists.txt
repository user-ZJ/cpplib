cmake_minimum_required(VERSION 3.14)



include_directories(
		 )
link_directories(
)


include(local_library)
mbase_local_library(opus opus.h)

add_library(opuswrapper OBJECT OpusWrapper.cpp)
target_link_libraries(opuswrapper opus myutil)
