/*
 * @Author: zack 
 * @Date: 2022-09-21 14:36:59 
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 14:37:35
 */
#ifndef BASE_TIMER_UITL_H_
#define BASE_TIMER_UITL_H_

#include <chrono>
#include <sstream>

namespace BASE_NAMESPACE
{

class Timer {
 public:
  Timer() : time_start_(std::chrono::steady_clock::now()) {}
  void Reset() { time_start_ = std::chrono::steady_clock::now(); }
  // return int in milliseconds
  int Elapsed() const {
    auto time_now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(time_now -
                                                                 time_start_)
        .count();
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> time_start_;
};

inline long getTimeStamp() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration_since_epoch =  now.time_since_epoch();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration_since_epoch);
  return ms.count();
}

inline std::string getSTimeStamp() {
  auto now = std::chrono::high_resolution_clock::now();
  std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

inline std::string getYear() {
  auto now = std::chrono::high_resolution_clock::now();
  // 转换为time_t类型
  std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
  // 转换为tm结构体
  struct tm now_tm;
  gmtime_r(&now_time_t, &now_tm);
  std::stringstream ss;
  ss<<(1900 + now_tm.tm_year);
  return ss.str();
}

}; // namespace DMAI

#endif
