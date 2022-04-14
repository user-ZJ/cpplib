/*
 * @Author: zack
 * @Date: 2021-09-03 11:26:57
 * @Last Modified by:   zack
 * @Last Modified time: 2021-09-03 11:26:57
 */

#pragma once
#include "string-util.h"
#include <boost/xpressive/xpressive.hpp>
#include <string>

namespace xpressive = boost::xpressive;

namespace BASE_NAMESPACE { namespace REGEX {

// static std::string PATT_HOUR_NUM24 = R"((?<!\d)(?:0?\d|1\d|20|21|22|23|24|00)(?!\d))";
// static std::string PATT_MINUTE = R"((?<!\d)[012345]\d(?!\d))";

// static std::string PATT_YEAR4 = R"([12]\d{3})";
// static std::string PATT_MONTH = R"(12|11|10|0?[987654321])";
// static std::string PATT_DAY = R"(31|30|[21]\d|0?[987654321])";

inline xpressive::wsregex to_wregex(const std::wstring &patt) {
  static xpressive::wsregex_compiler wcompiler;
  static std::map<size_t, xpressive::wsregex> wreg_cache;
  static std::hash<std::string> wh;
  size_t key = wh(to_string(patt));
  if (wreg_cache.count(key) == 0) { wreg_cache[key] = wcompiler.compile(patt); }
  return wreg_cache[key];
}

inline xpressive::sregex to_regex(const std::string &patt) {
  static xpressive::sregex_compiler compiler;
  static std::map<size_t, xpressive::sregex> reg_cache;
  static std::hash<std::string> h;
  size_t key = h(patt);
  if (reg_cache.count(key) == 0) { reg_cache[key] = compiler.compile(patt); }
  return reg_cache[key];
}

inline bool match(const std::string &s, const std::string &patt, std::vector<std::string> *tuple = nullptr) {
  /**
   * @brief 字符串是否完全匹配正则表达式
   *  patt 正则表达式的raw string
   *  s 待匹配的字符串
   *  tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   *  @return bool 是否完全匹配
   */
  xpressive::sregex sre = to_regex(patt);
  xpressive::smatch m;
  bool ret = xpressive::regex_match(s, m, sre);
  if (ret && tuple != nullptr) {
    for (auto i = 1; i < m.size(); ++i) { tuple->push_back(m[i].str()); }
  }
  return ret;
}

inline bool match(const std::wstring &s, const std::wstring &patt, std::vector<std::wstring> *tuple = nullptr) {
  /**
   * @brief 字符串是否完全匹配正则表达式
   *  patt 正则表达式的raw string
   *  s 待匹配的字符串
   *  tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   *  @return bool 是否完全匹配
   */
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsmatch m;
  bool ret = xpressive::regex_match(s, m, sre);
  if (ret && tuple != nullptr) {
    for (auto i = 1; i < m.size(); ++i) { tuple->push_back(m[i].str()); }
  }
  return ret;
}

inline bool search(const std::string &s, const std::string &patt, std::vector<std::string> *tuple = nullptr,
                   std::string *prefix = nullptr, std::string *suffix = nullptr) {
  /**
   * @brief 查找第一次匹配正则表达式的文本段
   * s 待匹配的字符串
   * patt 正则表达式的raw string
   * tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   * @return bool 是否找到匹配的文本段
   */
  xpressive::sregex sre = to_regex(patt);
  xpressive::smatch m;
  bool ret = xpressive::regex_search(s, m, sre);
  if (ret) {
    if (tuple != nullptr) {
      for (auto i = 1; i < m.size(); ++i) { tuple->push_back(m[i].str()); }
    }
    if (prefix != nullptr) (*prefix) = m.prefix().str();
    if (suffix != nullptr) (*suffix) = m.suffix().str();
  }
  return ret;
}

inline bool search(const std::wstring &s, const std::wstring &patt, std::vector<std::wstring> *tuple = nullptr,
                   std::wstring *prefix = nullptr, std::wstring *suffix = nullptr) {
  /**
   * @brief 查找第一次匹配正则表达式的文本段
   * s 待匹配的字符串
   * patt 正则表达式的raw string
   * tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   * @return bool 是否找到匹配的文本段
   */
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsmatch m;
  bool ret = xpressive::regex_search(s, m, sre);
  if (ret) {
    if (tuple != nullptr) {
      for (auto i = 1; i < m.size(); ++i) { tuple->push_back(m[i].str()); }
    }
    if (prefix != nullptr) (*prefix) = m.prefix().str();
    if (suffix != nullptr) (*suffix) = m.suffix().str();
  }
  return ret;
}

inline bool searchAll(const std::string &s, const std::string &patt,
                      std::vector<std::vector<std::string>> *tuple = nullptr) {
  /**
   * @brief 查找所有匹配正则表达式的文本段
   * s 待匹配的字符串
   * patt 正则表达式的raw string
   * tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   * @return bool 是否找到匹配的文本段
   */
  xpressive::sregex sre = to_regex(patt);
  xpressive::sregex_iterator cur(s.begin(), s.end(), sre), end;
  bool ret;
  for (; cur != end; ++cur) {
    ret = true;
    const xpressive::smatch &m = *cur;
    if (tuple != nullptr) {
      std::vector<std::string> t;
      for (auto i = 1; i < m.size(); i++) { t.push_back(m[i].str()); }
      tuple->emplace_back(t);
    }
  }
  return ret;
}

inline bool searchAll(const std::wstring &s, const std::wstring &patt,
                      std::vector<std::vector<std::wstring>> *tuple = nullptr) {
  /**
   * @brief 查找所有匹配正则表达式的文本段
   * s 待匹配的字符串
   * patt 正则表达式的raw string
   * tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   * @return bool 是否找到匹配的文本段
   */
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsregex_iterator cur(s.begin(), s.end(), sre), end;
  bool ret;
  for (; cur != end; ++cur) {
    ret = true;
    const xpressive::wsmatch &m = *cur;
    if (tuple != nullptr) {
      std::vector<std::wstring> t;
      for (auto i = 1; i < m.size(); i++) { t.push_back(m[i].str()); }
      tuple->emplace_back(t);
    }
  }
  return ret;
}

inline std::string replace(const std::string &s, const std::string &patt, const std::string &repl) {
  /**
   * @brief 替换第一次匹配的子串
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @param repl 替换的字符串，支持$n的方式使用分组匹配结果
   * @return 替换后的字符串
   */
  xpressive::sregex sre = to_regex(patt);
  return xpressive::regex_replace(s, sre, repl, xpressive::regex_constants::format_first_only);
}

inline std::wstring replace(const std::wstring &s, const std::wstring &patt, const std::wstring &repl) {
  /**
   * @brief 替换第一次匹配的子串
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @param repl 替换的字符串，支持$n的方式使用分组匹配结果
   * @return 替换后的字符串
   */
  xpressive::wsregex sre = to_wregex(patt);
  return xpressive::regex_replace(s, sre, repl, xpressive::regex_constants::format_first_only);
}

template <typename F>
inline std::string replaceLambda(const std::string &s, const std::string &patt, F &&lambda) {
  /**
   * @brief 替换第一次匹配的子串
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @param lambda 替换的字符串的lambda表达式,如：
   * [](const xpressive::smatch &m) -> std::string { return m[1].str() + "name=" + m[2].str(); }
   * @return 替换后的字符串
   */
  xpressive::sregex sre = to_regex(patt);
  return xpressive::regex_replace(s, sre, lambda, xpressive::regex_constants::format_first_only);
}

template <typename F>
inline std::wstring replaceLambda(const std::wstring &s, const std::wstring &patt, F &&lambda) {
  /**
   * @brief 替换第一次匹配的子串
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @param lambda 替换的字符串的lambda表达式,如：
   * [](const xpressive::smatch &m) -> std::string { return m[1].str() + "name=" + m[2].str(); }
   * @return 替换后的字符串
   */
  xpressive::wsregex sre = to_wregex(patt);
  return xpressive::regex_replace(s, sre, lambda, xpressive::regex_constants::format_first_only);
}

inline std::string replaceAll(const std::string &s, const std::string &patt, const std::string &repl) {
  /**
   * @brief 替换正则表达式匹配的子串
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @param repl 替换的字符串，支持$n的方式使用分组匹配结果
   * @return 替换后的字符串
   */
  xpressive::sregex sre = to_regex(patt);
  return xpressive::regex_replace(s, sre, repl);
}

inline std::wstring replaceAll(const std::wstring &s, const std::wstring &patt, const std::wstring &repl) {
  /**
   * @brief 替换正则表达式匹配的子串
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @param repl 替换的字符串，支持$n的方式使用分组匹配结果
   * @return 替换后的字符串
   */
  xpressive::wsregex sre = to_wregex(patt);
  return xpressive::regex_replace(s, sre, repl);
}

template <typename F>
inline std::string replaceAllLambda(const std::string &s, const std::string &patt, F &&lambda) {
  /**
   * @brief 替换正则表达式匹配的子串
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @param lambda 替换的字符串的lambda表达式,如：
   * [](const xpressive::smatch &m) -> std::string { return m[1].str() + "name=" + m[2].str(); }
   * @return 替换后的字符串
   */
  xpressive::sregex sre = to_regex(patt);
  return xpressive::regex_replace(s, sre, lambda);
}

template <typename F>
inline std::wstring replaceAllLambda(const std::wstring &s, const std::wstring &patt, F &&lambda) {
  /**
   * @brief 替换正则表达式匹配的子串
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @param lambda 替换的字符串的lambda表达式,如：
   * [](const xpressive::smatch &m) -> std::string { return m[1].str() + "name=" + m[2].str(); }
   * @return 替换后的字符串
   */
  xpressive::wsregex sre = to_wregex(patt);
  return xpressive::regex_replace(s, sre, lambda);
}

inline std::string replaceByDic(const std::string &text, const std::map<std::string, std::string> &dic) {
  auto result = text;
  for (const auto &elem : dic) { result = replaceAll(result, elem.first, elem.second); }
  return result;
}

inline std::wstring replaceByDic(std::wstring &text, const std::map<std::wstring, std::wstring> &dic) {
  auto result = text;
  for (const auto &elem : dic) { result = replaceAll(result, elem.first, elem.second); }
  return result;
}

inline std::vector<std::string> split(const std::string &s, const std::string &patt = R"(\s+)") {
  /**
   * @brief 使用正则表达式匹配分割符对字符串进行分割
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @return 分割后的字符串数组
   */
  std::vector<std::string> result;
  xpressive::sregex sre = to_regex(patt);
  xpressive::sregex_token_iterator begin(s.begin(), s.end(), sre,-1), end;
  for (auto iter = begin; iter != end; iter++) { result.push_back(iter->str()); }
  return result;
}

inline std::vector<std::wstring> split(const std::wstring &s, const std::wstring &patt = LR"(\s+)") {
  /**
   * @brief 使用正则表达式匹配分割符对字符串进行分割
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @return 分割后的字符串数组
   */
  std::vector<std::wstring> result;
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsregex_token_iterator begin(s.begin(), s.end(), sre,-1), end;
  for (auto iter = begin; iter != end; iter++) { result.push_back(iter->str()); }
  return result;
}

}}  // namespace BASE_NAMESPACE::REGEX
