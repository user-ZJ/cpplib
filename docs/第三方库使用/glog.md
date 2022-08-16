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

## 指定日志存储路径

1. 在代码中添加
   
   ```cpp
   #include <glog/logging.h>
   
   int main(int argc, char* argv[]) {
       // Initialize Google’s logging library.
       FLAGS_log_dir=/path/to/your/logdir
       google::InitGoogleLogging(argv[0]);
   
       // ...
       LOG(INFO) << "Found " << num_cookies << " cookies";
   }
   ```

2. 命令行参数,需要再本地安装glog
   
   ```shell
   ./your_application --log_dir=/some/log/directory
   ```

3. 环境变量，在未安装glog的时候使用
   
   ```shell
   GLOG_log_dir=/some/log/directory ./your_application
   ```

## 日志文件自动清理

```cpp
#include <glog/logging.h>

int main(int argc, char* argv[]) {
    // Initialize Google’s logging library.
    FLAGS_log_dir=/path/to/your/logdir
    google::EnableLogCleaner(3); // keep your logs for 3 days
    google::InitGoogleLogging(argv[0]);

    // ...
    LOG(INFO) << "Found " << num_cookies << " cookies";
}
```
