# 查找 sox 与 sonic（myutil、remove_noise 等音频模块依赖），并创建同名 IMPORTED target，
# 已有的 target_link_libraries(... sox sonic) 无需改动即可带上头文件目录和库的完整路径。
#   Ubuntu: apt install libsox-dev libsonic-dev
#   macOS:  brew install sox
#           sonic 需从源码安装，编译和安装都要带 PREFIX，否则 dylib 的 install name 仍指向 /usr/local：
#             git clone https://github.com/waywardgeek/sonic.git && cd sonic
#             make PREFIX=$(brew --prefix) && make install PREFIX=$(brew --prefix)
# 装在其他位置时，可通过 -DCMAKE_PREFIX_PATH=<prefix> 指定。

# Apple Silicon 的 Homebrew 前缀 /opt/homebrew 不在 CMake 和编译器的默认搜索路径中
set(AUDIO_DEPS_PREFIXES /opt/homebrew /usr/local)

function(import_audio_library name header)
  string(TOUPPER ${name} upper)
  find_path(${upper}_INCLUDE_DIR ${header} PATHS ${AUDIO_DEPS_PREFIXES} PATH_SUFFIXES include)
  find_library(${upper}_LIBRARY NAMES ${name} PATHS ${AUDIO_DEPS_PREFIXES} PATH_SUFFIXES lib)
  if(NOT ${upper}_INCLUDE_DIR OR NOT ${upper}_LIBRARY)
    message(FATAL_ERROR "${name} not found (${upper}_INCLUDE_DIR=${${upper}_INCLUDE_DIR}, "
                        "${upper}_LIBRARY=${${upper}_LIBRARY}); see cmake/sox.cmake for how to install")
  endif()
  add_library(${name} UNKNOWN IMPORTED)
  set_target_properties(${name} PROPERTIES
    IMPORTED_LOCATION "${${upper}_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${${upper}_INCLUDE_DIR}")
  message(STATUS "Found ${name}: ${${upper}_LIBRARY}")
endfunction()

import_audio_library(sox sox.h)
import_audio_library(sonic sonic.h)
