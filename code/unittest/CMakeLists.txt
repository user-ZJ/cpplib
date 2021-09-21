cmake_minimum_required(VERSION 3.14)
project(test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
link_directories(
		)

set(TEST_SOURCES path_unittest.cpp)
add_executable(path_test ${TEST_SOURCES})
target_link_libraries(path_test mbase gtest_main)
add_test(NAME path_test COMMAND path_test)

include(GoogleTest)
gtest_discover_tests(path_test)
