if(NOT DEFINED boost_SOURCE_DIR)
FetchContent_Declare(boost
  # 按顺序尝试，SourceForge 会跳转到就近镜像
  URL      https://sourceforge.net/projects/boost/files/boost/1.87.0/boost_1_87_0.tar.gz/download
           https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz
  # SourceForge 链接不以 .tar.gz 结尾，需显式指定文件名才能识别解压格式
  DOWNLOAD_NAME boost_1_87_0.tar.gz
  URL_HASH SHA256=f55c340aa49763b1925ccf02b2e83f35fdcf634c9d5164a2acb87540173c741d
)
FetchContent_MakeAvailable(boost)
endif()

include_directories(${boost_SOURCE_DIR})

if(MSVC)
  add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
endif()
