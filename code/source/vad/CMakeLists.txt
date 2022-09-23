cmake_minimum_required(VERSION 3.14)
project(vadwrapper)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
link_directories(
)


add_library(vadwrapper FVadWrapper.cpp)
target_link_libraries(vadwrapper fvad utils)

