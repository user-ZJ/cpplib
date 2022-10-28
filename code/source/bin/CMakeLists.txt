cmake_minimum_required(VERSION 3.14)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )


add_executable(aes_crypto AESCrypto.cpp)
target_link_libraries(aes_crypto crypto_wrapper)

