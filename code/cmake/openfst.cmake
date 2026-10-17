if(NOT ANDROID)
  # 使用本工程 glog.cmake / gflags.cmake 构建的 glog 和 gflags，保证与主工程是同一份且平台一致
  set(OPENFST_DEP_CPPFLAGS "-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR} -DGLOG_USE_GLOG_EXPORT")
  set(OPENFST_GFLAGS_LIB_DIR $<TARGET_FILE_DIR:gflags_nothreads_static>)
  set(OPENFST_GLOG_LIB_DIR $<TARGET_LINKER_FILE_DIR:glog>)
  # configure 会运行测试程序，需能找到共享库 glog
  set(OPENFST_DEP_LDFLAGS "-L${OPENFST_GFLAGS_LIB_DIR} -L${OPENFST_GLOG_LIB_DIR} -Wl,-rpath,${OPENFST_GLOG_LIB_DIR}")

  set(CONFIG_FLAGS "")
  if(NOT FST_HAVE_BIN)
    if(MSVC)
      set(HAVE_BIN OFF CACHE BOOL "Build the fst binaries" FORCE)
    else()
      set(CONFIG_FLAGS "--disable-bin")
    endif()
  endif()

  # The original openfst uses GNU Build System to run configure and build.
  # So, we use "OpenFST port for Windows" to build openfst with cmake in Windows.
  # Openfst is compiled with glog/gflags to avoid log and flag conflicts with log and flags in wenet/libtorch.
  # To build openfst with gflags and glog, we comment out some vars of {flags, log}.h and flags.cc.
  set(openfst_SOURCE_DIR ${fc_base}/openfst-src CACHE PATH "OpenFST source directory")
  set(openfst_BINARY_DIR ${fc_base}/openfst-build CACHE PATH "OpenFST build directory")
  set(openfst_PREFIX_DIR ${fc_base}/openfst-subbuild/openfst-populate-prefix CACHE PATH "OpenFST prefix directory")
  if(NOT MSVC)
    ExternalProject_Add(openfst
      URL               https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.5.tar.gz
      URL_HASH          SHA256=70d9e795b9bb3eb157908a2fdde17d5d1c1be35ab21e208187a4f230291606b1
      PREFIX            ${openfst_PREFIX_DIR}
      SOURCE_DIR        ${openfst_SOURCE_DIR}
      BINARY_DIR        ${openfst_BINARY_DIR}
      PATCH_COMMAND     ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/patch/fix_openfst.cmake
      CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure ${CONFIG_FLAGS} --prefix=${openfst_PREFIX_DIR} --enable-static=yes --enable-shared=no
                          # configure 在检测编译器前就给 CXXFLAGS 赋了值，autoconf 因此不会默认加 -O2
                          "CXXFLAGS=-O2"
                          "CPPFLAGS=${OPENFST_DEP_CPPFLAGS} ${TORCH_CXX_FLAGS}"
                          "LDFLAGS=${OPENFST_DEP_LDFLAGS}"
                          "LIBS=-lgflags_nothreads -lglog -lpthread"
      COMMAND           ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
      # $(MAKE) 继承外层 make 的 -j 与 jobserver
      BUILD_COMMAND     $(MAKE)
    )
    add_dependencies(openfst gflags_nothreads_static glog)
    link_directories(${openfst_PREFIX_DIR}/lib)
  else()
    add_compile_options(/W0 /wd4244 /wd4267)
    FetchContent_Declare(openfst
      URL           https://github.com/kkm000/openfst/archive/refs/tags/win/1.6.5.1.tar.gz
      URL_HASH      SHA256=02c49b559c3976a536876063369efc0e41ab374be1035918036474343877046e
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(openfst)
    # add_dependencies(fst gflags glog)
    target_link_libraries(fst PUBLIC gflags_nothreads_static glog)
  endif()
  include_directories(${openfst_SOURCE_DIR}/src/include)
else()
  set(openfst_BINARY_DIR ${build_DIR}/wenet-openfst-android-1.0.1.aar/jni)
  include_directories(${openfst_BINARY_DIR}/include)
  link_directories(${openfst_BINARY_DIR}/${ANDROID_ABI})
  link_libraries(log gflags_nothreads glog fst)
endif()
