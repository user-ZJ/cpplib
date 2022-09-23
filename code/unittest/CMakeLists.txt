cmake_minimum_required(VERSION 3.14)
project(test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
link_directories(
		)

aux_source_directory(. UNITTEST_SRC)
add_executable(unit_test ${UNITTEST_SRC})
target_link_libraries(unit_test utils gtest_main)
add_test(NAME unit_test COMMAND unit_test)

# include(GoogleTest)
# gtest_discover_tests(path_test)
