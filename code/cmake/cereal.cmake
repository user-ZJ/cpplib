set(SKIP_PERFORMANCE_COMPARISON ON)
set(SKIP_PERFORMANCE_COMPARISON ON)
set(BUILD_SANDBOX OFF)
set(BUILD_DOC OFF)
FetchContent_Declare(cereal
  URL https://github.com/user-ZJ/cereal/archive/refs/tags/v1.3.2-beta.3.zip
  URL_HASH SHA256=ae4ecd5c5da502a8d1c76d6a4c5cfe339f27ecb7dd687f8256501305c065f5a2
)
FetchContent_MakeAvailable(cereal)
include_directories(${cereal_SOURCE_DIR}/include)