if(NOT DEFINED minizip_SOURCE_DIR)
FetchContent_Declare(minizip
  GIT_REPOSITORY    https://github.com/zlib-ng/minizip-ng.git
  GIT_TAG   99d39015e29703af2612277180ea586805f655ea
)
FetchContent_MakeAvailable(minizip)
endif()
include_directories(${minizip_SOURCE_DIR})
link_directories(${minizip_BINARY_DIR})