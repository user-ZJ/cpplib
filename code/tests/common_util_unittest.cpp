#include <map>
#include <string>
#include <unordered_map>
#include "utils/common-util.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(CommonUtil, map) {
  std::map<std::string, int> dic{{"a", 1}};
  EXPECT_EQ(1, getFromDic(dic, std::string("a"), 0));
  EXPECT_EQ(0, getFromDic(dic, std::string("missing"), 0));
}

TEST(CommonUtil, unorderedMap) {
  std::unordered_map<std::string, int> dic{{"a", 2}};
  EXPECT_EQ(2, getFromDic(dic, std::string("a"), -1));
  EXPECT_EQ(-1, getFromDic(dic, std::string("missing"), -1));
}
