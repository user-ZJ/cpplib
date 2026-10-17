# 从源码编译 rnnoise，并创建 IMPORTED target rnnoise，已有的 target_link_libraries(... rnnoise) 无需改动。
# 固定在 xiph/rnnoise v0.2 之前的最后一个提交（2021-03-12）：头文件和 libtool 版本（.so.0.4.1）
# 与原 3rdparty/rnnoise 预编译库一致；v0.2 起更换了模型格式。
set(RNNOISE_COMMIT 7f449bf8bd3b933891d12c30112268c4090e4d59)
set(rnnoise_SOURCE_DIR ${fc_base}/rnnoise-src)
set(rnnoise_BINARY_DIR ${fc_base}/rnnoise-build)
set(rnnoise_PREFIX_DIR ${fc_base}/rnnoise-subbuild/rnnoise-populate-prefix)
set(rnnoise_LIBRARY ${rnnoise_PREFIX_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}rnnoise${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Add(rnnoise_ext
  URL               https://github.com/xiph/rnnoise/archive/${RNNOISE_COMMIT}.tar.gz
  URL_HASH          SHA256=05396f221b308099491d9f4838b58a71ba2224a5e4a1c5276de6385abf369131
  PREFIX            ${rnnoise_PREFIX_DIR}
  SOURCE_DIR        ${rnnoise_SOURCE_DIR}
  BINARY_DIR        ${rnnoise_BINARY_DIR}
  INSTALL_DIR       ${rnnoise_PREFIX_DIR}
  PATCH_COMMAND     ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/patch/rnnoise/CMakeLists.txt <SOURCE_DIR>/CMakeLists.txt
  CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                    -DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
  BUILD_BYPRODUCTS  ${rnnoise_LIBRARY}
)

# IMPORTED target 的头文件目录必须在配置阶段就存在
file(MAKE_DIRECTORY ${rnnoise_PREFIX_DIR}/include)
add_library(rnnoise STATIC IMPORTED)
set_target_properties(rnnoise PROPERTIES
  IMPORTED_LOCATION "${rnnoise_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${rnnoise_PREFIX_DIR}/include")
if(NOT WIN32)
  set_property(TARGET rnnoise PROPERTY INTERFACE_LINK_LIBRARIES m)
endif()
add_dependencies(rnnoise rnnoise_ext)
