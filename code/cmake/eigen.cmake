FetchContent_Declare(eigen
  URL    https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
  URL_HASH SHA256=7985975b787340124786f092b3a07d594b2e9cd53bbfe5f3d9b1daee7d55f56f
)
FetchContent_Populate(eigen)
include_directories(${eigen_SOURCE_DIR})