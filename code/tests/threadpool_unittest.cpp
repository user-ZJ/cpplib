#include <chrono>
#include <string>
#include "utils/threadpool.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(ThreadPool, enqueue) {
  ThreadPool pool(2);
  auto sum = pool.enqueue([](int x, int y) { return x + y; }, 20, 22);
  auto text = pool.enqueue([] { return std::string("ok"); });
  EXPECT_EQ(42, sum.get());
  EXPECT_EQ("ok", text.get());
}

TEST(ThreadPool, runsTasks) {
  ThreadPool pool(2);
  std::vector<std::future<int>> results;
  for (int i = 0; i < 8; ++i) {
    results.push_back(pool.enqueue([i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      return i * i;
    }));
  }
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(i * i, results[i].get());
  }
}
