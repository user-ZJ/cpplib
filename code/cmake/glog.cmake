# if(NOT DEFINED glog_SOURCE_DIR)
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/refs/tags/v0.6.0.zip
  URL_HASH SHA256=122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022
)
FetchContent_MakeAvailable(glog)
# endif()
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
link_directories(${glog_BINARY_DIR})
