cmake_minimum_required(VERSION 3.14)
project(test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


include_directories(
	../source
	# /data/code/libtorch/include/
	# /data/code/libtorch/include/torch/csrc/api/include
		 )
link_directories(
	# /home/dmai/code/pytorch/build_libtorch/build/lib
		)

# set(Torch_DIR /data/code/libtorch/share/cmake/Torch)
# find_package(Torch REQUIRED)

add_executable(path_util_test path_test.cpp)
target_link_libraries(path_util_test myutil)

add_executable(ctensor_test ctensor_test.cpp)
target_link_libraries(ctensor_test myutil)


add_executable(audio_test audio_test.cpp)
target_link_libraries(audio_test PUBLIC myutil)

add_executable(redis_test redis_test.cpp)
target_link_libraries(redis_test dbwrapper)

add_executable(mongodb_test mongodb_test.cpp)
target_link_libraries(mongodb_test dbwrapper)

add_executable(rapidjson_test rapidjson_test.cpp)


add_executable(threadpool_test threadpool_test.cpp)
target_link_libraries(threadpool_test myutil pthread)


add_executable(crypto_test crypto_test.cpp)
target_link_libraries(crypto_test crypto_wrapper)

add_executable(string_util_test string_util_test.cpp)
target_link_libraries(string_util_test myutil)

add_executable(regex_util_test regex_util_test.cpp)
target_link_libraries(regex_util_test myutil)

add_executable(eigen_test eigen_test.cpp)
target_link_libraries(eigen_test)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wfatal-errors -fpermissive")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wfatal-errors -fpermissive")
# add_executable(minizip_test minizip_test.cpp)
# target_link_libraries(minizip_test myutil)


add_executable(filesystem_test filesystem_test.cpp)
target_link_libraries(filesystem_test)

FIND_PACKAGE(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
add_executable(omp_test omp_test.cpp)
target_link_libraries(omp_test myutil)



add_executable(vad_test vad_test.cpp)
target_link_libraries(vad_test vadwrapper sox myutil)

add_executable(rnnoise_test rnnoise_test.cpp)
target_link_libraries(rnnoise_test  remove_noise sox myutil)

add_executable(mysql_test mysql_test.cpp)
target_link_libraries(mysql_test dbwrapper glog gflags)



add_executable(just_test just_test.cpp)
target_link_libraries(just_test crypto ssl myutil)