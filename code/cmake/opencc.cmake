if(NOT DEFINED opencc_SOURCE_DIR)
#set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(opencc
  GIT_REPOSITORY    https://github.com/BYVoid/OpenCC.git
  GIT_TAG ver.1.1.4
)
FetchContent_Populate(opencc)
include(${opencc_SOURCE_DIR}/CMakeLists.txt)
endif()
include_directories(${opencc_SOURCE_DIR}/src)
link_directories(${opencc_BINARY_DIR})