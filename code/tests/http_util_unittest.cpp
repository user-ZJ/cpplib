#include <sstream>
#include <string>
#include "web/HttpUtil.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(HttpUtil, readLimited) {
  std::istringstream in("hello");
  std::string out;
  ReadLimited(in, out, 5);
  EXPECT_EQ("hello", out);

  std::istringstream too_big("hello");
  EXPECT_THROW(ReadLimited(too_big, out, 4), HttpError);

  std::istringstream empty;
  ReadLimited(empty, out, 0);
  EXPECT_TRUE(out.empty());
}

TEST(HttpUtil, parseJsonObject) {
  auto obj = ParseJsonObject(R"({"name":"ada"})");
  EXPECT_EQ("ada", RequireString(obj, "name"));

  try {
    ParseJsonObject("[]");
    FAIL() << "array should be rejected";
  } catch (const HttpError &e) {
    EXPECT_EQ(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST, e.status());
  }

  try {
    ParseJsonObject("{");
    FAIL() << "broken json should be rejected";
  } catch (const HttpError &e) {
    EXPECT_EQ(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST, e.status());
  }

  try {
    RequireString(ParseJsonObject("{}"), "name");
    FAIL() << "missing field should be rejected";
  } catch (const HttpError &e) {
    EXPECT_EQ(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST, e.status());
    EXPECT_NE(std::string::npos, std::string(e.what()).find("name"));
  }
}
