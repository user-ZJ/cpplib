cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
link_directories(
)


add_library(opuswrapper OBJECT OpusWrapper.cpp)
target_link_libraries(opuswrapper opus myutil)

