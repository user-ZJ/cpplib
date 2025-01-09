if(NOT DEFINED boost_SOURCE_DIR)
FetchContent_Declare(boost
  URL      https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz
  URL_HASH SHA256=f55c340aa49763b1925ccf02b2e83f35fdcf634c9d5164a2acb87540173c741d
)
FetchContent_MakeAvailable(boost)
endif()

include_directories(${boost_SOURCE_DIR})

if(MSVC)
  add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
endif()
