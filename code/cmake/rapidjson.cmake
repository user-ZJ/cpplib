if(NOT DEFINED rapidjson_SOURCE_DIR)
# FetchContent_Declare(rapidjson
#   GIT_REPOSITORY    https://github.com/Tencent/rapidjson.git
#   GIT_TAG   v1.1.0
# )
FetchContent_Declare(rapidjson
  URL      https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz
  URL_HASH SHA256=bf7ced29704a1e696fbccf2a2b4ea068e7774fa37f6d7dd4039d0787f8bed98e
)
FetchContent_Populate(rapidjson)
endif()
include_directories(${rapidjson_SOURCE_DIR}/include)
