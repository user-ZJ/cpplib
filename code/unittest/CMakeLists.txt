cmake_minimum_required(VERSION 3.14)

aux_source_directory(. UNITTEST_SRC)
add_executable(unit_test ${UNITTEST_SRC})
target_include_directories(unit_test PRIVATE ${CMAKE_SOURCE_DIR}/source/utils)
target_link_libraries(unit_test PRIVATE myutil crypto_wrapper gtest Poco::Net Poco::JSON Poco::Foundation)
add_test(NAME unit_test COMMAND $<TARGET_FILE:unit_test>)
set_tests_properties(unit_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
