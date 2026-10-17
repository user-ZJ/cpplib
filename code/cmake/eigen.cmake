if(NOT DEFINED eigen_SOURCE_DIR)
FetchContent_Declare(eigen
  URL    https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
  URL_HASH SHA256=b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2
)
FetchContent_Populate(eigen)
endif()
include_directories(${eigen_SOURCE_DIR})
