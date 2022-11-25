# third_party: cppjieba
if(NOT DEFINED jieba_SOURCE_DIR)
FetchContent_Declare(jieba
	URL      https://github.com/yanyiwu/cppjieba/archive/refs/tags/v5.0.3.tar.gz
	URL_HASH SHA256=b40848a553dab24d7fcdb6dbdea2486102212baf58466d1c3c3481381af91248
)
FetchContent_Populate(jieba)
endif()
set(jiaba_INCLUDE_DIR ${jieba_SOURCE_DIR}/include ${jieba_SOURCE_DIR}/deps)
include_directories(${jiaba_INCLUDE_DIR})
link_directories(${jieba_BINARY_DIR})
