/*
 * @Author: zack
 * @Date: 2021-10-05 10:28:12
 * @Last Modified by: zack
 * @Last Modified time: 2021-10-05 10:30:43
 */
#ifndef BASE_STRING_UTIL_H_
#define BASE_STRING_UTIL_H_

#include <algorithm>
#include <cassert>
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

namespace BASE_NAMESPACE {

const char WHITESPACE[] = " \n\r\t\f\v";

template <typename T>
std::string v2s(const std::vector<T> &v) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < v.size(); ++i) {
    ss << v[i];
    if (i != v.size() - 1) { ss << ", "; }
  }
  ss << "]";
  return ss.str();
}

template <typename Container>
std::string printCollection(const Container &c) {
  std::stringstream ss;
  if (c.empty()) {
    ss << "(empty)";
  } else {
    typename Container::const_iterator itr = c.begin();
    ss << "[ " << *itr++;
    while (itr != c.end())
      ss << ", " << *itr++;
    ss << " ]";
  }
  return ss.str();
}

// wstring和string之间转换
using convert_t = std::codecvt_utf8<wchar_t>;
static std::wstring_convert<convert_t, wchar_t> strconverter;

inline std::string to_string(std::wstring wstr) {
  // if need debug,use this code
  // std::locale lc("zh_CN.UTF-8");
  // std::locale::global(lc);
  // std::wcout<<wstr;
  return strconverter.to_bytes(wstr);
}

inline std::wstring to_wstring(std::string str) {
  return strconverter.from_bytes(str);
}

// float转换为string并修正格式
inline std::string toFormatStr(float number, int precision = 2) {
  // std::ostringstream oss;
  // oss << std::setprecision(7) << number;
  // return oss.str();
  char buffer[64];
  memset(buffer, 0, sizeof(buffer));
  std::string format = "%." + std::to_string(precision) + "f";
  snprintf(buffer, sizeof(buffer), format.c_str(), number);
  return std::string(buffer);
}

inline std::vector<std::string> splitStringToVector(const std::string &full, const char *delim = " ") {
  size_t start = 0, found = 0, end = full.size();
  std::vector<std::string> out;
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    if (found != start && start != end) { out.push_back(full.substr(start, found - start)); }
    start = found + 1;
  }
  return out;
}

inline std::vector<std::wstring> splitStringToVector(const std::wstring &full, const wchar_t *delim = L" ") {
  size_t start = 0, found = 0, end = full.size();
  std::vector<std::wstring> out;
  while (found != std::wstring::npos) {
    found = full.find_first_of(delim, start);
    if (found != start && start != end) { out.push_back(full.substr(start, found - start)); }
    start = found + 1;
  }
  return out;
}

inline void SplitUTF8StringToChars(const std::string &str, std::vector<std::string> *chars) {
  chars->clear();
  int bytes = 1;
  for (size_t i = 0; i < str.length(); i += bytes) {
    assert((str[i] & 0xF8) <= 0xF0);
    if ((str[i] & 0x80) == 0x00) {
      // The first 128 characters (US-ASCII) in UTF-8 format only need one byte.
      bytes = 1;
    } else if ((str[i] & 0xE0) == 0xC0) {
      // The next 1,920 characters need two bytes to encode,
      // which covers the remainder of almost all Latin-script alphabets.
      bytes = 2;
    } else if ((str[i] & 0xF0) == 0xE0) {
      // Three bytes are needed for characters in the rest of
      // the Basic Multilingual Plane, which contains virtually all characters
      // in common use, including most Chinese, Japanese and Korean characters.
      bytes = 3;
    } else if ((str[i] & 0xF8) == 0xF0) {
      // Four bytes are needed for characters in the other planes of Unicode,
      // which include less common CJK characters, various historic scripts,
      // mathematical symbols, and emoji (pictographic symbols).
      bytes = 4;
    }
    chars->push_back(str.substr(i, bytes));
  }
}

inline std::string joinVectorToString(const std::vector<std::string> &vec, const char *delim = " ") {
  std::string res = "";
  for (size_t i = 0; i < vec.size(); i++) {
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
  for (size_t i = 0; i < vec.size(); i++) {
    if (i == vec.size() - 1) {
      res += vec[i];
    } else {
      res = res + vec[i] + delim;
    }
  }
  return res;
}

inline int UTF8StringLength(const std::string &str) {
  int len = 0;
  int bytes = 1;
  for (size_t i = 0; i < str.length(); i += bytes) {
    if ((str[i] & 0x80) == 0x00) {
      bytes = 1;
    } else if ((str[i] & 0xE0) == 0xC0) {
      bytes = 2;
    } else if ((str[i] & 0xF0) == 0xE0) {
      bytes = 3;
    } else if ((str[i] & 0xF8) == 0xF0) {
      bytes = 4;
    }
    ++len;
  }
  return len;
}

inline bool CheckEnglishChar(const std::string &ch) {
  // all english characters should be encoded in one byte
  if (ch.size() != 1) return false;
  // english words may contain apostrophe, i.e., "He's"
  return isalpha(ch[0]) || ch[0] == '\'';
}

inline bool CheckEnglishWord(const std::string &word) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(word, &chars);
  for (size_t k = 0; k < chars.size(); k++) {
    if (!CheckEnglishChar(chars[k])) { return false; }
  }
  return true;
}

inline bool CheckNumChar(const std::string& ch) {
  if (ch.size() != 1) return false;
  return (ch[0]>='0') and (ch[0] <='9');
}


inline std::string Ltrim(const std::string &str) {
  size_t start = str.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : str.substr(start);
}

inline std::string Rtrim(const std::string &str) {
  size_t end = str.find_last_not_of(WHITESPACE);
  return (end == std::string::npos) ? "" : str.substr(0, end + 1);
}

inline std::string Trim(const std::string &str) {
  return Rtrim(Ltrim(str));
}

inline std::string strip(const std::string &str) {
  static char white=' ',enter='\n';
  if(str.empty())
    return str;
  size_t strBegin=0, strEnd=str.size()-1, strRange;
  while(strBegin<strEnd){
    if(str[strBegin]==white || str[strBegin]==enter)
      ++strBegin;
    else
      break;
  }
  while(strBegin<strEnd){
    if(str[strEnd]==white || str[strEnd]==enter)
      --strEnd;
    else
      break;
  }
  strRange = strEnd - strBegin + 1;
  return str.substr(strBegin, strRange);
}

inline std::wstring strip(const std::wstring &str) {
  static wchar_t white=L' ',enter=L'\n';
  if(str.empty())
    return str;
  size_t strBegin=0, strEnd=str.size()-1, strRange;
  while(strBegin<strEnd){
    if(str[strBegin]==white || str[strBegin]==enter)
      ++strBegin;
    else
      break;
  }
  while(strBegin<strEnd){
    if(str[strEnd]==white || str[strEnd]==enter)
      --strEnd;
    else
      break;
  }
  strRange = strEnd - strBegin + 1;
  return str.substr(strBegin, strRange);
}


// 获取系统环境变量
inline std::string getEnvVar(const std::string &key) {
  if (key.empty()) return std::string("");
  char *val = getenv(key.c_str());
  return val == NULL ? std::string("") : std::string(val);
}
// 增加/修改系统环境变量
inline int setEnvVar(const std::string &key, const std::string &value) {
  if (key.empty()) return -1;
  return setenv(key.c_str(), value.c_str(), 1);
}
// 清空环境变量
inline int unsetEnvVar(const std::string &key) {
  if (key.empty()) return -1;
  return unsetenv(key.c_str());
  ;
}

inline bool endswith(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

inline bool endswith(std::wstring const &fullString, std::wstring const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

inline bool startswith(std::string const &fullString, std::string const &starts) {
  if (fullString.length() >= starts.length()) {
    return (0 == fullString.compare(0, starts.length(), starts));
  } else {
    return false;
  }
}

inline bool startswith(std::wstring const &fullString, std::wstring const &starts) {
  if (fullString.length() >= starts.length()) {
    return (0 == fullString.compare(0, starts.length(), starts));
  } else {
    return false;
  }
}

inline std::string toLowercase(const std::string &str) {
  std::string s = str;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

inline std::wstring toLowercase(const std::wstring &str) {
  std::wstring s = str;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

inline std::string toUppercase(const std::string &str) {
  std::string s = str;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
  return s;
}

inline std::wstring toUppercase(const std::wstring &str) {
  std::wstring s = str;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
  return s;
}

inline bool haveAlpha(const std::string &str) {
  for (auto &c : str) {
    if (isalpha(c)) return true;
  }
  return false;
}

inline bool haveAlpha(const std::wstring &wstr) {
  std::string str = to_string(wstr);
  for (auto &c : str) {
    if (isalpha(c)) return true;
  }
  return false;
}

};  // namespace BASE_NAMESPACE

#endif
