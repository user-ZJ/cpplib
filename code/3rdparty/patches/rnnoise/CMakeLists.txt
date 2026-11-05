# 由 cmake/rnnoise.cmake 在 patch 阶段复制到 rnnoise 源码根目录。
# 该版本上游只提供 autotools 构建；这里按其 Makefile.am / configure.ac 中库的编译方式用 CMake 构建，
# 免去对 autoconf / automake / libtool 的依赖。
cmake_minimum_required(VERSION 3.10)
project(rnnoise C)

add_library(rnnoise STATIC
  src/denoise.c
  src/rnn.c
  src/rnn_data.c
  src/rnn_reader.c
  src/pitch.c
  src/kiss_fft.c
  src/celt_lpc.c)
target_include_directories(rnnoise PUBLIC include)
# RNNOISE_BUILD 使头文件中的 RNNOISE_EXPORT 生效，配合 hidden 可见性只导出公开 API
target_compile_definitions(rnnoise PRIVATE RNNOISE_BUILD)
set_target_properties(rnnoise PROPERTIES
  C_VISIBILITY_PRESET hidden
  # 会被链接进共享库 remove_noise
  POSITION_INDEPENDENT_CODE ON)

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
  target_link_libraries(rnnoise PRIVATE ${MATH_LIBRARY})
endif()

install(TARGETS rnnoise ARCHIVE DESTINATION lib)
install(FILES include/rnnoise.h DESTINATION include)
