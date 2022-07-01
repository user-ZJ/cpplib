FetchContent_Declare(rapidjson
  GIT_REPOSITORY    https://github.com/Tencent/rapidjson.git
  GIT_TAG   v1.1.0
)
FetchContent_Populate(rapidjson)
include_directories(${rapidjson_SOURCE_DIR}/include)