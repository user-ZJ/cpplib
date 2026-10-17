if(NOT DEFINED flags_SOURCE_DIR)
FetchContent_Declare(gflags
  URL      https://github.com/gflags/gflags/archive/refs/tags/v2.3.1.zip
  URL_HASH SHA256=d08fb50ffe6fffb7e895f72ef0c160059fd88a676978bd20c69f2299384c8e03
)
FetchContent_MakeAvailable(gflags)
endif()
include_directories(${gflags_BINARY_DIR}/include)
link_directories(${gflags_BINARY_DIR})
