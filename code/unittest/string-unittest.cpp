#include<iostream>
#include<string>
#include "string-util.h"
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