cuda_add_executable(test_dense test_dense.cpp)
target_link_libraries(test_dense culayer cubase ${CUDA_LIBRARIES} cudnn)

cuda_add_executable(test_init_kernel test_init_kernel.cpp)
target_link_libraries(test_init_kernel cukernel cubase ${CUDA_LIBRARIES})

cuda_add_executable(test_matrix_transpose_kernel test_matrix_transpose_kernel.cpp)
target_link_libraries(test_matrix_transpose_kernel cukernel cubase ${CUDA_LIBRARIES})