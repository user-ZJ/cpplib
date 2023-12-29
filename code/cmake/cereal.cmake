set(SKIP_PERFORMANCE_COMPARISON ON)
set(SKIP_PERFORMANCE_COMPARISON ON)
set(BUILD_SANDBOX OFF)
set(BUILD_DOC OFF)
FetchContent_Declare(cereal
  URL https://github.com/user-ZJ/cereal/archive/refs/tags/v1.3.2-beta.0.zip
  URL_HASH SHA256=e8f9fd2576d421447133c8f2e462bc905c572e4a957e5d23eafc88679a4893c6
)
FetchContent_MakeAvailable(cereal)
include_directories(${cereal_SOURCE_DIR}/include)