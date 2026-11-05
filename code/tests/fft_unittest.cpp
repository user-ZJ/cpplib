#include <cmath>
#include <vector>
#include "utils/fft.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(Fft, tables) {
  constexpr int n = 8;
  std::vector<float> sintbl(n + n / 4);
  std::vector<int> bitrev(n);
  make_sintbl(n, sintbl.data());
  make_bitrev(n, bitrev.data());
  EXPECT_FLOAT_EQ(0.f, sintbl[0]);
  EXPECT_FLOAT_EQ(1.f, sintbl[n / 4]);

  std::vector<int> seen(n, 0);
  for (int i = 0; i < n; ++i) {
    ASSERT_GE(bitrev[i], 0);
    ASSERT_LT(bitrev[i], n);
    seen[bitrev[i]] += 1;
  }
  for (int count : seen) {
    EXPECT_EQ(1, count);
  }
}

TEST(Fft, forwardAndInverse) {
  constexpr int n = 8;
  std::vector<float> sintbl(n + n / 4, 0.f);
  std::vector<int> bitrev(n, 0);
  make_sintbl(n, sintbl.data());
  make_bitrev(n, bitrev.data());

  std::vector<float> x{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f};
  std::vector<float> y(n, 0.f);
  auto original = x;
  EXPECT_EQ(0, fft(bitrev.data(), sintbl.data(), x.data(), y.data(), n));
  EXPECT_EQ(0, fft(bitrev.data(), sintbl.data(), x.data(), y.data(), -n));
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(original[i], x[i], 1e-4f);
    EXPECT_NEAR(0.f, y[i], 1e-4f);
  }
}
