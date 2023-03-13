#include <chrono>
#include <iostream>
#include <vector>

#include "utils/threadpool.h"

using namespace BASE_NAMESPACE;

int foo(int i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Task " << i << " completed" << std::endl;
    return i*i;
}

int main() {

  ThreadPool pool(4);
  std::vector<std::future<int>> results;

  for (int i = 0; i < 8; ++i) {
    results.emplace_back(pool.enqueue([i] {
      std::cout << "hello " << i << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "world " << i << std::endl;
      return i * i;
    }));
  }

  for (auto &&result : results)
    std::cout << result.get() << "\n";
  std::cout << std::endl;

  results.clear();
  for (int i = 0; i < 8; ++i) {
    results.emplace_back(pool.enqueue(foo,i));
  }

  for (auto &&result : results)
    std::cout << "result:"<<result.get() << "\n";
  std::cout << std::endl;

  return 0;
}
