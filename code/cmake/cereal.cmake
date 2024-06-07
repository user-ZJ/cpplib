set(SKIP_PERFORMANCE_COMPARISON ON)
set(SKIP_PERFORMANCE_COMPARISON ON)
set(BUILD_SANDBOX OFF)
set(BUILD_DOC OFF)
FetchContent_Declare(cereal
  URL https://github.com/user-ZJ/cereal/archive/refs/tags/v1.3.2-beta.5.zip
  URL_HASH SHA256=fedfab6ae9248ebd7c57744c2831219435a3b6bf499a8da9131b84a12490a3c1
)
FetchContent_MakeAvailable(cereal)
include_directories(${cereal_SOURCE_DIR}/include)