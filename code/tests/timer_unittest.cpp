#include <chrono>
#include <ctime>
#include <iomanip>
#include <regex>
#include <thread>
#include "utils/timer.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(Timer, elapsedAndReset) {
  Timer timer;
  int before = timer.Elapsed();
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  int after = timer.Elapsed();
  EXPECT_GT(after, before);
  timer.Reset();
  EXPECT_LT(timer.Elapsed(), after);
}

TEST(Timer, stamps) {
  auto first = getTimeStamp();
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_GE(getTimeStamp(), first);

  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  EXPECT_EQ(std::to_string(1900 + utc.tm_year), getYear());
  EXPECT_TRUE(std::regex_match(getSTimeStamp(), std::regex(R"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")));
}
