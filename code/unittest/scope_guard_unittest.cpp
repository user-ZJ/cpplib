#include <utility>
#include "utils/ScopeGuard.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(ScopeGuardTest, runsOnExit) {
  int n = 0;
  {
    ScopeGuard guard([&] { n = 1; });
  }
  EXPECT_EQ(1, n);
}

TEST(ScopeGuardTest, dismissSkipsCallback) {
  int n = 0;
  {
    ScopeGuard guard([&] { n = 1; });
    guard.dismiss();
  }
  EXPECT_EQ(0, n);
}
