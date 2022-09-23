googletest笔记
=======================

示例
--------------

.. code-block:: cpp

	//cat example.cc
	#include<gtest/gtest.h>
	TEST(Mytest, failing_test){
		EXPECT_TRUE(false);
	}


.. code-block:: cmake

	# CMakeLists.txt
	include(FetchContent)
	set(FETCHCONTENT_QUIET off)
	set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/thirdpart)

	FetchContent_Declare(googletest
		URL      https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
		URL_HASH SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5
	)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	set(GTEST_HAS_PTHREAD 1)
	FetchContent_MakeAvailable(googletest) #后续链接gtest_main即可
	enable_testing()

	add_executable(example example.cpp)
	target_link_libraries(example gtest_main)

	include(GoogleTest)
	gtest_discover_tests(hello_test)


执行：

.. code-block:: shell
	
	$ ./example
	# 或者运行./ctest执行所有用例
	$ ctest


