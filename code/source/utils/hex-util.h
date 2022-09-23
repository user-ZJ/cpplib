/*
 * @Author: zack 
 * @Date: 2022-09-21 15:17:32 
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 15:18:00
 */
#ifndef BASE_HEX_UTIL_H_
#define BASE_HEX_UTIL_H_
#include <string>
#include <vector>

namespace BASE_NAMESPACE {

static const int eof = std::char_traits<char>::eof();
static const char digits[] = "0123456789abcdef0123456789ABCDEF";

std::string HexBinaryEncoder(const std::vector<char> &binary, bool isUpper = false) {
  int uppercase = isUpper ? 16 : 0;
  std::string res = "";
  for (const auto &c : binary) {
    res += digits[uppercase + ((c >> 4) & 0xF)];
    res += digits[_uppercase + (c & 0xF)];
  }
  return res;
}

std::vector<char> HexBinaryDecoder(const std::string &str) {
  std::vector<char> res;
  char c, n;
  for (int i=0;i<str.length()/2;i++) {
    n = str[2*i];
    if (n >= '0' && n <= '9')
      c = n - '0';
    else if (n >= 'A' && n <= 'F')
      c = n - 'A' + 10;
    else if (n >= 'a' && n <= 'f')
      c = n - 'a' + 10;
    c <<= 4;
    if (n >= '0' && n <= '9')
		c |= n - '0';
	else if (n >= 'A' && n <= 'F')
		c |= n - 'A' + 10;
	else if (n >= 'a' && n <= 'f')
		c |= n - 'a' + 10;
    res.push_back(c);
  }
  return res;
}

};  // namespace BASE_NAMESPACE

#endif