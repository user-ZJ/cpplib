#include <cstring>
#include <string>
#include <vector>
#include "utils/uuid_wrapper.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(Uuid, format) {
  auto first = gen_uuid();
  auto second = gen_uuid();
  EXPECT_EQ(36u, first.size());
  EXPECT_EQ('-', first[8]);
  EXPECT_EQ('-', first[13]);
  EXPECT_EQ('-', first[18]);
  EXPECT_EQ('-', first[23]);
  EXPECT_NE(first, second);

  auto hex = gen_uuid_hex();
  EXPECT_EQ(32u, hex.size());
  EXPECT_EQ(std::string::npos, hex.find('-'));
  for (char c : hex) {
    EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) << c;
  }
}
