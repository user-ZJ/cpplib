cmake_minimum_required(VERSION 3.14)

add_executable(unit_test
    base64_unittest.cpp
    common_util_unittest.cpp
    container_unittest.cpp
    crypto_unittest.cpp
    fft_unittest.cpp
    file_util_unittest.cpp
    hex_unittest.cpp
    http_util_unittest.cpp
    main.cpp
    ndtensor_unittest.cpp
    path_unittest.cpp
    regex_unittest.cpp
    scope_guard_unittest.cpp
    string-unittest.cpp
    threadpool_unittest.cpp
    timer_unittest.cpp
    uuid_unittest.cpp
)
target_include_directories(unit_test PRIVATE ${CMAKE_SOURCE_DIR}/source/utils)
target_link_libraries(unit_test PRIVATE myutil crypto_wrapper GTest::gtest Poco::Net Poco::JSON Poco::Foundation)
add_test(NAME unit_test COMMAND $<TARGET_FILE:unit_test>)
set_tests_properties(unit_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
if(TARGET kaldi-matrix)
    target_sources(unit_test PRIVATE kaldi_matrix_unittest.cpp)
    target_link_libraries(unit_test PRIVATE kaldi-matrix)
endif()
