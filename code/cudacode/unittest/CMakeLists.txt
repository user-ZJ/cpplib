cuda_add_executable(test_dense test_dense.cpp)
target_link_libraries(test_dense culayer cubase ${CUDA_LIBRARIES} cudnn)