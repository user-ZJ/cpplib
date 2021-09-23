#pragma once

#include <codecvt>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace DMAI {

using convert_t = std::codecvt_utf8<wchar_t>;
static std::wstring_convert<convert_t, wchar_t> strconverter;

inline std::string to_string(std::wstring wstr) {
  //if need debug,use this code
  //std::locale lc("zh_CN.UTF-8");
  //std::locale::global(lc);
  // std::wcout<<wstr;
  return strconverter.to_bytes(wstr);
}

inline std::wstring to_wstring(std::string str) {
  return strconverter.from_bytes(str);
}

//float转换为string并修正格式
inline std::string toFormatStr(float number) {
  std::ostringstream oss;
  oss << std::setprecision(7) << number;
  return oss.str();
}

inline std::vector<std::string> splitStringToVector(const std::string &full, const char *delim = " ") {
  size_t start = 0, found = 0, end = full.size();
  std::vector<std::string> out;
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    if (found != start && start != end) {
      out.push_back(full.substr(start, found - start));
    }
    start = found + 1;
  }
  return out;
}

inline std::string joinVectorToString(const std::vector<std::string> &vec, const char *delim = " ") {
  std::string res = "";
  for (int i = 0; i < vec.size(); i++) {
    if (i == vec.size() - 1) {
      res += vec[i];
    } else {
      res = res + vec[i] + delim;
    }
  }
  return res;
}

inline std::wstring joinVectorToString(const std::vector<std::wstring> &vec, const wchar_t *delim = L" ") {
  std::wstring res = L"";
  for (int i = 0; i < vec.size(); i++) {
    if (i == vec.size() - 1) {
      res += vec[i];
    } else {
      res = res + vec[i] + delim;
    }
  }
  return res;
}

};  // namespace DMAI
