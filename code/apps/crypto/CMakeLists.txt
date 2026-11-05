add_executable(aes_crypto AESCrypto.cpp)
target_link_libraries(aes_crypto PRIVATE crypto_wrapper)
