/*
 * @Author: zack 
 * @Date: 2022-09-21 14:36:59 
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 14:37:35
 */
#ifndef BASE_TIMER_UITL_H_
#define BASE_TIMER_UITL_H_

#include <chrono>

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

}; // namespace BASE_NAMESPACE

#endif
