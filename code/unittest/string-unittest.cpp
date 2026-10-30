#include<iostream>
#include<string>
#include "utils/string-util.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(StringTest, toLowercase){
  std::string str = "AbC";
  EXPECT_EQ("abc",toLowercase(str));
}

TEST(StringTest, toUppercase){
  std::string str = "AbC";
  EXPECT_EQ("ABC",toUppercase(str));
}

TEST(StringTest, trim) {
  EXPECT_EQ("ab", Trim(" \t ab\n"));
  EXPECT_EQ("ab", Ltrim("  ab"));
  EXPECT_EQ("ab", Rtrim("ab  "));
  EXPECT_EQ("", Trim(" \n\t"));
  EXPECT_EQ("a", strip(" \n a \n "));
}

TEST(StringTest, splitAndJoin) {
  auto parts = splitStringToVector("a  b c", " ");
  ASSERT_EQ(3u, parts.size());
  EXPECT_EQ("a", parts[0]);
  EXPECT_EQ("b", parts[1]);
  EXPECT_EQ("c", parts[2]);
  EXPECT_EQ("a-b", joinVectorToString({"a", "b"}, "-"));
  EXPECT_TRUE(splitStringToVector("").empty());
}

TEST(StringTest, prefixSuffix) {
  EXPECT_TRUE(startswith("hello", "he"));
  EXPECT_FALSE(startswith("hello", "hello!"));
  EXPECT_TRUE(endswith("hello", "lo"));
  EXPECT_FALSE(endswith("hello", "x"));
}

TEST(StringTest, utf8) {
  EXPECT_EQ(3, UTF8StringLength("abc"));
  EXPECT_EQ(2, UTF8StringLength("a你"));
  std::vector<std::string> chars;
  SplitUTF8StringToChars("a你", &chars);
  ASSERT_EQ(2u, chars.size());
  EXPECT_EQ("a", chars[0]);
  EXPECT_EQ("你", chars[1]);
}

TEST(StringTest, englishAndFormat) {
  EXPECT_TRUE(CheckEnglishWord("He's"));
  EXPECT_FALSE(CheckEnglishWord("hello!"));
  EXPECT_FALSE(CheckNumChar("a"));
  EXPECT_TRUE(CheckNumChar("7"));
  EXPECT_EQ("[1, 2]", v2s(std::vector<int>{1, 2}));
  EXPECT_EQ("(empty)", printCollection(std::vector<int>{}));
  EXPECT_EQ("1.50", toFormatStr(1.5f, 2));
  EXPECT_TRUE(haveAlpha("a1"));
  EXPECT_FALSE(haveAlpha("12"));
}

TEST(StringTest, envVar) {
  const std::string key = "MBASE_UT_ENV_KEY";
  EXPECT_EQ("", getEnvVar(""));
  EXPECT_EQ(-1, setEnvVar("", "x"));
  EXPECT_EQ(0, setEnvVar(key, "v"));
  EXPECT_EQ("v", getEnvVar(key));
  EXPECT_EQ(0, unsetEnvVar(key));
  EXPECT_EQ("", getEnvVar(key));
}