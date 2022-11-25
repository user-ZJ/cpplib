if(NOT DEFINED fvad_SOURCE_DIR)
FetchContent_Declare(fvad
  GIT_REPOSITORY    https://github.com/dpirch/libfvad.git
  GIT_TAG   847a37297a8ca3fe80c4d878a2003f2c5106b0bf
)
FetchContent_MakeAvailable(fvad)
endif()
include_directories(${fvad_SOURCE_DIR}/include)
link_directories(${fvad_BINARY_DIR})