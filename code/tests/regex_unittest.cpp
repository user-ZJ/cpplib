#include <string>
#include <vector>
#include "utils/regex-util.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE::REGEX;

TEST(RegexTest, match) {
  EXPECT_TRUE(match("abc", R"(abc)"));
  EXPECT_TRUE(match("abc", R"(a.c)"));
  EXPECT_FALSE(match("abc", R"(ab)"));
  EXPECT_FALSE(match("", R"(a)"));
  EXPECT_FALSE(match("abc", ""));
}

TEST(RegexTest, splitAndReplace) {
  auto parts = split("a  b\tc", R"(\s+)");
  ASSERT_EQ(3u, parts.size());
  EXPECT_EQ("a", parts[0]);
  EXPECT_EQ("b", parts[1]);
  EXPECT_EQ("c", parts[2]);
  EXPECT_EQ("axax", replaceAll("a1a2", R"(\d)", "x"));
  EXPECT_EQ("axa2", replace("a1a2", R"(\d)", "x"));
  EXPECT_EQ("keep", replaceAll("keep", "", "x"));
}

TEST(RegexTest, tokenAndSearch) {
  auto words = token("say Hello", R"([A-Za-z]+)");
  ASSERT_EQ(2u, words.size());
  EXPECT_EQ("say", words[0]);
  EXPECT_EQ("Hello", words[1]);

  std::vector<std::string> groups;
  std::string prefix;
  std::string suffix;
  EXPECT_TRUE(search("xx123yy", R"(\d+)", &groups, &prefix, &suffix));
  ASSERT_EQ(1u, groups.size());
  EXPECT_EQ("123", groups[0]);
  EXPECT_EQ("xx", prefix);
  EXPECT_EQ("yy", suffix);
}
