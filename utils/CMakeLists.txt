cmake_minimum_required(VERSION 2.8)
project(zack_utils)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")

include_directories(
		 )
link_directories(
		)


set(UTILS_SOURCES 
	path.cpp
	text-utils.cpp
	)


add_library(${PROJECT_NAME} SHARED ${UTILS_SOURCES})

target_link_libraries(${PROJECT_NAME}
		)
INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

set(TEST_SOURCES path_test.cpp)
add_executable(path_test ${TEST_SOURCES})
target_link_libraries(path_test zack_utils)
INSTALL(TARGETS path_test RUNTIME DESTINATION bin)

set(TEXT_TEST_SOURCES text-utils-test.cpp)
add_executable(text_test ${TEXT_TEST_SOURCES})
target_link_libraries(text_test zack_utils)
INSTALL(TARGETS text_test RUNTIME DESTINATION bin)
