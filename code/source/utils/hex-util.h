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

inline std::string HexBinaryEncoder(const std::vector<unsigned char> &binary, bool isUpper = false) {
  int uppercase = isUpper ? 16 : 0;
  std::string res = "";
  for (const auto &c : binary) {
    res += digits[uppercase + ((c >> 4) & 0xF)];
    res += digits[uppercase + (c & 0xF)];
  }
  return res;
}

inline int HexDigitValue(char n) {
  if (n >= '0' && n <= '9') return n - '0';
  if (n >= 'A' && n <= 'F') return n - 'A' + 10;
  if (n >= 'a' && n <= 'f') return n - 'a' + 10;
  return -1;
}

// 长度为奇数或含非十六进制字符时返回空
inline std::vector<unsigned char> HexBinaryDecoder(const std::string &str) {
  std::vector<unsigned char> res;
  if (str.size() % 2 != 0) return res;
  res.reserve(str.size() / 2);
  for (std::size_t i = 0; i < str.size(); i += 2) {
    int hi = HexDigitValue(str[i]);
    int lo = HexDigitValue(str[i + 1]);
    if (hi < 0 || lo < 0) return {};
    res.push_back(static_cast<unsigned char>((hi << 4) | lo));
  }
  return res;
}

};  // namespace BASE_NAMESPACE

#endif