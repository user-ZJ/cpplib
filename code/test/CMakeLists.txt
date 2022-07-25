cmake_minimum_required(VERSION 3.14)
project(test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


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


add_executable(audio_test audio_test.cpp)
target_link_libraries(audio_test sox vadwrapper)

add_executable(redis_test redis_test.cpp)
target_link_libraries(redis_test dbwrapper)

add_executable(mongodb_test mongodb_test.cpp)
target_link_libraries(mongodb_test dbwrapper)

add_executable(rapidjson_test rapidjson_test.cpp)


add_executable(threadpool_test threadpool_test.cpp)
target_link_libraries(threadpool_test mbase pthread)


# add_executable(crypto_test crypto_test.cpp)
# target_link_libraries(crypto_test cryptopp.a pthread)

add_executable(string_util_test string_util_test.cpp)
target_link_libraries(string_util_test mbase)

add_executable(regex_util_test regex_util_test.cpp)
target_link_libraries(regex_util_test mbase)

add_executable(eigen_test eigen_test.cpp)
target_link_libraries(eigen_test)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wfatal-errors -fpermissive")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wfatal-errors -fpermissive")
add_executable(minizip_test minizip_test.cpp)
target_link_libraries(minizip_test mbase)


add_executable(filesystem_test filesystem_test.cpp)
target_link_libraries(filesystem_test)