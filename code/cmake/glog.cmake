find_package(glog CONFIG REQUIRED)
if(NOT TARGET glog AND TARGET glog::glog)
    add_library(glog ALIAS glog::glog)
endif()
