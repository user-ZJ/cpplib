FetchContent_Declare(minizip
  GIT_REPOSITORY    https://github.com/zlib-ng/minizip-ng.git
  GIT_TAG   99d39015e29703af2612277180ea586805f655ea
)
FetchContent_MakeAvailable(minizip)
include_directories(${minizip_SOURCE_DIR})