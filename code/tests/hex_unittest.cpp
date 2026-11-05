#include <string>
#include <vector>
#include "utils/hex-util.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(HexTest, encode) {
  std::vector<unsigned char> bytes{0x0a, 0xff, 0x00};
  EXPECT_EQ("0aff00", HexBinaryEncoder(bytes));
  EXPECT_EQ("0AFF00", HexBinaryEncoder(bytes, true));
  EXPECT_EQ("", HexBinaryEncoder({}));
}

TEST(HexTest, decode) {
  auto bytes = HexBinaryDecoder("0aff00");
  ASSERT_EQ(3u, bytes.size());
  EXPECT_EQ(0x0a, bytes[0]);
  EXPECT_EQ(0xff, bytes[1]);
  EXPECT_EQ(0x00, bytes[2]);
  EXPECT_EQ("0aff00", HexBinaryEncoder(bytes));
}

TEST(HexTest, rejectInvalid) {
  EXPECT_TRUE(HexBinaryDecoder("abc").empty());
  EXPECT_TRUE(HexBinaryDecoder("zz").empty());
  EXPECT_TRUE(HexBinaryDecoder("0G").empty());
}
