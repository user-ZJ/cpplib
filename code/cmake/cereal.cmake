set(SKIP_PERFORMANCE_COMPARISON ON)
set(SKIP_PERFORMANCE_COMPARISON ON)
set(BUILD_SANDBOX OFF)
set(BUILD_DOC OFF)
FetchContent_Declare(cereal
  URL https://github.com/user-ZJ/cereal/archive/refs/tags/v1.3.2-beta.1.zip
  URL_HASH SHA256=73dbec95fbb84612c514831d5246419fc405c0bff6931957bb722bfdee78b737
)
FetchContent_MakeAvailable(cereal)
include_directories(${cereal_SOURCE_DIR}/include)