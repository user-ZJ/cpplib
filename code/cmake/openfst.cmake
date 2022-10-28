if(NOT ANDROID)
  set(gflags_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/gflags/include)
  set(glog_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/glog/include)

  set(gflags_BINARY_DIR ${CMAKE_SOURCE_DIR}/3rdparty/gflags/lib/Linux/x64)
  set(glog_BINARY_DIR ${CMAKE_SOURCE_DIR}/3rdparty/glog/lib/Linux/x64)


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
      CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure ${CONFIG_FLAGS} --prefix=${openfst_PREFIX_DIR} --enable-static=yes --enable-shared=no
                          "CPPFLAGS=-I${gflags_SOURCE_DIR} -I${glog_SOURCE_DIR} ${TORCH_CXX_FLAGS}"
                          "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                          "LIBS=-lgflags_nothreads -lglog -lpthread"
      COMMAND           ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
      BUILD_COMMAND     make -j$(nproc)
    )
    # add_dependencies(openfst gflags glog)
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
