cmake_minimum_required(VERSION 3.14)
project(test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
	../source
		 )
link_directories(
		)

set(TEST_SOURCES text-utils-test.cpp)
add_executable(text_test ${TEST_SOURCES})
target_link_libraries(text_test mbase)



add_executable(path_util_test path_test.cpp)
target_link_libraries(path_util_test mbase)

add_executable(ctensor_test ctensor_test.cpp)
target_link_libraries(ctensor_test mbase)


add_executable(threadpool_test threadpool_test.cpp)
target_link_libraries(threadpool_test mbase)


# add_executable(minizip_test minizip_test.cpp)
# target_link_libraries(minizip_test mbase minizip)