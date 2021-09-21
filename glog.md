# glog

## 示例

```cpp
//cat example.cc
#include <glog/logging.h>

int main(int argc, char* argv[]) {
    // Initialize Google’s logging library.
    google::InitGoogleLogging(argv[0]);

    // ...
    LOG(INFO) << "Found " << num_cookies << " cookies";
}
```

```cmake
include(FetchContent)
set(FETCHCONTENT_QUIET off)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/thirdpart)

# third_party: glog
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/refs/tags/v0.5.0.zip
  URL_HASH SHA256=21bc744fb7f2fa701ee8db339ded7dce4f975d0d55837a97be7d46e8382dea5a
)
set(WITH_CUSTOM_PREFIX ON)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

add_executable(example example.cpp)
target_link_libraries(example glog)
```

执行：

```shell
export GLOG_logtostderr=1
export GLOG_v=2
$ ./example
```

