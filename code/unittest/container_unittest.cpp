#include "utils/blocking_queue.h"
#include "utils/safestl/thread_safe_queue.h"
#include "gtest/gtest.h"

TEST(ThreadSafeQueue, pushPop) {
  thread_safe::queue<int> q;
  EXPECT_TRUE(q.empty());
  q.push(1);
  q.push(2);
  EXPECT_EQ(2u, q.size());
  EXPECT_EQ(1, q.front());
  EXPECT_EQ(2, q.back());
  q.pop();
  EXPECT_EQ(2, q.front());
  q.pop();
  EXPECT_TRUE(q.empty());
}

TEST(BlockingQueue, pushPop) {
  BASE_NAMESPACE::BlockingQueue<int> q(4);
  EXPECT_TRUE(q.Empty());
  q.Push(1);
  q.Push(2);
  EXPECT_EQ(2u, q.Size());
  EXPECT_EQ(1, q.Pop());
  EXPECT_EQ(2, q.Pop());
  EXPECT_TRUE(q.Empty());
}
