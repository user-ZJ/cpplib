if(NOT DEFINED opencc_SOURCE_DIR)
#set(BUILD_SHARED_LIBS OFF)
# FetchContent_Declare(opencc
#   GIT_REPOSITORY    https://github.com/BYVoid/OpenCC.git
#   GIT_TAG ver.1.1.4
# )
FetchContent_Declare(opencc
  URL      https://github.com/BYVoid/OpenCC/archive/refs/tags/ver.1.1.6.tar.gz
  URL_HASH SHA256=169bff4071ffe814dc16df7d180ff6610db418f4816e9c0ce02cf874bdf058df
)
FetchContent_Populate(opencc)
include(${opencc_SOURCE_DIR}/CMakeLists.txt)
endif()
include_directories(${opencc_SOURCE_DIR}/src)
# target_compile_options(opencc PRIVATE -DSTATIC_LIBRARY)
# set_directory_properties(PROPERTIES BUILD_SHARED_LIBS OFF)
link_directories(${opencc_BINARY_DIR})