#include <cctype>
#include <string>
#include "utils/base64-util.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(Base64Test, roundtrip) {
  const std::string text = "hello";
  auto encoded = base64_encode(reinterpret_cast<const unsigned char *>(text.data()), text.size());
  EXPECT_EQ("aGVsbG8=", encoded);
  EXPECT_EQ(text, base64_decode(encoded));
}

TEST(Base64Test, empty) {
  EXPECT_EQ("", base64_encode(nullptr, 0));
  EXPECT_EQ("", base64_decode(""));
}

TEST(Base64Test, padding) {
  const std::string text = "hell";
  auto encoded = base64_encode(reinterpret_cast<const unsigned char *>(text.data()), text.size());
  EXPECT_EQ("aGVsbA==", encoded);
  EXPECT_EQ(text, base64_decode(encoded));
}
