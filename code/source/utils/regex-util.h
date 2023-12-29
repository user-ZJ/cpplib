/*
 * @Author: zack
 * @Date: 2021-09-03 11:26:57
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 15:14:31
 */

#ifndef BASE_REGEX_UTIL_H_
#define BASE_REGEX_UTIL_H_
#include "utils/logging.h"
#include "utils/string-util.h"
#include <boost/xpressive/xpressive.hpp>
#include <string>
#include<mutex>


namespace xpressive = boost::xpressive;

namespace BASE_NAMESPACE { namespace REGEX {

// static std::string PATT_HOUR_NUM24 = R"((?<!\d)(?:0?\d|1\d|20|21|22|23|24|00)(?!\d))";
// static std::string PATT_MINUTE = R"((?<!\d)[012345]\d(?!\d))";

// static std::string PATT_YEAR4 = R"([12]\d{3})";
// static std::string PATT_MONTH = R"(12|11|10|0?[987654321])";
// static std::string PATT_DAY = R"(31|30|[21]\d|0?[987654321])";

//中文标点
//。 ？ ！ ， 、 ； ： “ ” ‘ ' （ ） 《 》 〈 〉 【 】 『 』 「 」 ﹃ ﹄ 〔 〕 … — ～ ﹏ ￥ • ·
//[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5|\u2022|\u00b7]
static const std::string ZHPunct =
  R"([\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5|\u2022|\u00b7])";
static const std::string ENPunct =
  R"([\.\?!,;:'"\(\)\[\]\{\}<>/\-\\$@`~#%^\*_\+=\|])";         // 英文标点.?!,;:'"(){}<>/-\$@`~#%^+=|
static const std::string ZHWord = R"([\u4e00-\u9fa5])";        // 中文汉字
static const std::string ZHTWord = R"([\u3400-\u4dbf])";       // 中文繁体
static const std::string ENWord = R"([a-zA-Z][A-Za-z_\-']*)";  // 英文单词

static std::unordered_map<std::wstring,xpressive::wsregex> wregexs_map;
static std::unordered_map<std::string,xpressive::sregex> regexs_map;
static std::mutex regex_mutex;

inline xpressive::wsregex to_wregex(const std::wstring &patt) {
  static xpressive::wsregex_compiler wcompiler;
  xpressive::wsregex my_regex;
  try {
    if(wregexs_map.count(patt)){
      return wregexs_map[patt];
    }else{
      std::unique_lock<std::mutex> lck(regex_mutex);
      my_regex = wcompiler.compile(patt);
      wregexs_map[patt] = my_regex;
    }
  }
  catch (...) {
    LOG(ERROR) << "compile regex error:" << to_string(patt);
    return xpressive::wsregex();
  }
  return my_regex;
}

inline xpressive::sregex to_regex(const std::string &patt) {
  static xpressive::sregex_compiler compiler;
  xpressive::sregex my_regex;
  try {
    if(regexs_map.count(patt)){
      return regexs_map[patt];
    }else{
      std::unique_lock<std::mutex> lck(regex_mutex);
      my_regex = compiler.compile(patt);
      regexs_map[patt] = my_regex;
    }
  }
  catch (...) {
    LOG(ERROR) << "compile regex error:" << patt;
    return xpressive::sregex();
  }
  return my_regex;
}

inline bool match(const std::string &s, const std::string &patt, std::vector<std::string> *tuple = nullptr) {
  /**
   * @brief 字符串是否完全匹配正则表达式
   *  patt 正则表达式的raw string
   *  s 待匹配的字符串
   *  tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   *  @return bool 是否完全匹配
   */
  if (s.empty() or patt.empty()) return false;
  if (tuple != nullptr) tuple->clear();
  xpressive::sregex sre = to_regex(patt);
  xpressive::smatch m;
  bool ret = xpressive::regex_match(s, m, sre);
  if (ret && tuple != nullptr) {
    for (auto i = 0; i < m.size(); ++i) {
      tuple->push_back(m[i].str());
    }
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
  if (s.empty() or patt.empty()) return false;
  if (tuple != nullptr) tuple->clear();
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsmatch m;
  bool ret = xpressive::regex_match(s, m, sre);
  if (ret && tuple != nullptr) {
    for (auto i = 0; i < m.size(); ++i) {
      tuple->push_back(m[i].str());
    }
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
  if (s.empty() or patt.empty()) return false;
  if (tuple != nullptr) tuple->clear();
  xpressive::sregex sre = to_regex(patt);
  xpressive::smatch m;
  bool ret = xpressive::regex_search(s, m, sre);
  if (ret) {
    if (tuple != nullptr) {
      for (auto i = 0; i < m.size(); ++i) {
        tuple->push_back(m[i].str());
      }
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
  if (s.empty() or patt.empty()) return false;
  if (tuple != nullptr) tuple->clear();
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsmatch m;
  bool ret = xpressive::regex_search(s, m, sre);
  if (ret) {
    if (tuple != nullptr) {
      for (auto i = 0; i < m.size(); ++i) {
        tuple->push_back(m[i].str());
      }
    }
    if (prefix != nullptr) (*prefix) = m.prefix().str();
    if (suffix != nullptr) (*suffix) = m.suffix().str();
  }
  return ret;
}

inline bool searchAll(const std::string &s, const std::string &patt,
                      std::vector<std::vector<std::string>> *tuple = nullptr,
                      std::vector<std::string> *prefix = nullptr, std::vector<std::string> *suffix = nullptr) {
  /**
   * @brief 查找所有匹配正则表达式的文本段
   * s 待匹配的字符串
   * patt 正则表达式的raw string
   * tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   * @return bool 是否找到匹配的文本段
   */
  if (s.empty() or patt.empty()) return false;
  if (tuple != nullptr) tuple->clear();
  if (prefix != nullptr) prefix->clear();
  if (suffix != nullptr) suffix->clear();
  xpressive::sregex sre = to_regex(patt);
  xpressive::sregex_iterator cur(s.begin(), s.end(), sre), end;
  bool ret;
  for (; cur != end; ++cur) {
    ret = true;
    const xpressive::smatch &m = *cur;
    if (tuple != nullptr) {
      std::vector<std::string> t;
      for (auto i = 0; i < m.size(); i++) {
        t.push_back(m[i].str());
      }
      tuple->emplace_back(t);
    }
    if (prefix != nullptr) { prefix->push_back(m.prefix().str()); }
    if (suffix != nullptr) { suffix->push_back(m.suffix().str()); }
  }
  return ret;
}

inline bool searchAll(const std::wstring &s, const std::wstring &patt,
                      std::vector<std::vector<std::wstring>> *tuple = nullptr,
                      std::vector<std::wstring> *prefix = nullptr, std::vector<std::wstring> *suffix = nullptr) {
  /**
   * @brief 查找所有匹配正则表达式的文本段
   * s 待匹配的字符串
   * patt 正则表达式的raw string
   * tuple 正则匹配返回的元组，null时表示不获取元组返回结果
   * @return bool 是否找到匹配的文本段
   */
  if (s.empty() or patt.empty()) return false;
  if (tuple != nullptr) tuple->clear();
  if (prefix != nullptr) prefix->clear();
  if (suffix != nullptr) suffix->clear();
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsregex_iterator cur(s.begin(), s.end(), sre), end;
  bool ret;
  for (; cur != end; ++cur) {
    ret = true;
    const xpressive::wsmatch &m = *cur;
    if (tuple != nullptr) {
      std::vector<std::wstring> t;
      for (auto i = 0; i < m.size(); i++) {
        t.push_back(m[i].str());
      }
      tuple->emplace_back(t);
    }
    if (prefix != nullptr) { prefix->push_back(m.prefix().str()); }
    if (suffix != nullptr) { suffix->push_back(m.suffix().str()); }
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
  if (s.empty() or patt.empty()) return s;
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
  if (s.empty() or patt.empty()) return s;
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
  if (s.empty() or patt.empty()) return s;
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
   * [](const xpressive::wsmatch &m) -> std::wstring { return m[1].str() + "name=" + m[2].str(); }
   * @return 替换后的字符串
   */
  if (s.empty() or patt.empty()) return s;
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
  if (s.empty() or patt.empty()) return s;
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
  if (s.empty() or patt.empty()) return s;
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
  if (s.empty() or patt.empty()) return s;
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
   * [](const xpressive::wsmatch &m) -> std::wstring { return m[1].str() + "name=" + m[2].str(); }
   * @return 替换后的字符串
   */
  if (s.empty() or patt.empty()) return s;
  xpressive::wsregex sre = to_wregex(patt);
  return xpressive::regex_replace(s, sre, lambda);
}

inline std::string replaceByDic(const std::string &text, const std::map<std::string, std::string> &dic) {
  auto result = text;
  for (const auto &elem : dic) {
    result = replaceAll(result, elem.first, elem.second);
  }
  return result;
}

inline std::string replaceByDic(const std::string &text, const std::unordered_map<std::string, std::string> &dic) {
  auto result = text;
  for (const auto &elem : dic) {
    result = replaceAll(result, elem.first, elem.second);
  }
  return result;
}

inline std::wstring replaceByDic(std::wstring &text, const std::map<std::wstring, std::wstring> &dic) {
  auto result = text;
  for (const auto &elem : dic) {
    result = replaceAll(result, elem.first, elem.second);
  }
  return result;
}

inline std::wstring replaceByDic(std::wstring &text, const std::unordered_map<std::wstring, std::wstring> &dic) {
  auto result = text;
  for (const auto &elem : dic) {
    result = replaceAll(result, elem.first, elem.second);
  }
  return result;
}

inline std::vector<std::string> split(const std::string &s, const std::string &patt = R"(\s+)") {
  /**
   * @brief 使用正则表达式匹配分割符对字符串进行分割
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @return 分割后的字符串数组
   */
  if (patt.empty()) return {s};
  std::vector<std::string> result;
  xpressive::sregex sre = to_regex(patt);
  xpressive::sregex_token_iterator begin(s.begin(), s.end(), sre, -1), end;
  for (auto iter = begin; iter != end; iter++) {
    result.push_back(iter->str());
  }
  return result;
}

inline std::vector<std::wstring> split(const std::wstring &s, const std::wstring &patt = LR"(\s+)") {
  /**
   * @brief 使用正则表达式匹配分割符对字符串进行分割
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @return 分割后的字符串数组
   */
  if (patt.empty()) return {s};
  std::vector<std::wstring> result;
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsregex_token_iterator begin(s.begin(), s.end(), sre, -1), end;
  for (auto iter = begin; iter != end; iter++) {
    result.push_back(iter->str());
  }
  return result;
}

inline std::vector<std::string> token(const std::string &s, const std::string &patt = R"([a-zA-Z][A-Za-z_']*)") {
  /**
   * @brief 使用正则表达式匹配,提取满足条件的文本
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @return 提取的字符串数组
   */
  if (patt.empty()) return {};
  std::vector<std::string> result;
  xpressive::sregex sre = to_regex(patt);
  xpressive::sregex_token_iterator begin(s.begin(), s.end(), sre), end;
  for (auto iter = begin; iter != end; iter++) {
    result.push_back(iter->str());
  }
  return result;
}

inline std::vector<std::wstring> token(const std::wstring &s,
                                       const std::wstring &patt = LR"([\u4e00-\u9fa5]|[a-zA-Z][A-Za-z_']*)") {
  /**
   * @brief 使用正则表达式匹配,提取满足条件的文本
   * @param s 原始字符串
   * @param patt 正则表达式的raw string
   * @return 提取的字符串数组
   */
  if (patt.empty()) return {};
  std::vector<std::wstring> result;
  xpressive::wsregex sre = to_wregex(patt);
  xpressive::wsregex_token_iterator begin(s.begin(), s.end(), sre), end;
  for (auto iter = begin; iter != end; iter++) {
    result.push_back(iter->str());
  }
  return result;
}

inline bool isZHPunct(const std::string &text){
  std::wstring wtext = to_wstring(text);
  return match(wtext,to_wstring(ZHPunct+"+"));
}

inline bool isENPunct(const std::string &text){
  std::wstring wtext = to_wstring(text);
  return match(wtext,to_wstring(ENPunct+"+"));
}

inline bool isPunct(const std::string &text){
  return isENPunct(text) || isZHPunct(text);
}

inline bool isZHWord(const std::string &text){
  std::wstring wtext = to_wstring(text);
  bool is_zh_word = match(wtext,to_wstring(ZHWord+"+"));
  bool is_zht_word = match(wtext,to_wstring(ZHTWord+"+"));
  return is_zh_word || is_zht_word;
}

inline bool isENWord(const std::string &text){
  std::wstring wtext = to_wstring(text);
  return match(wtext,to_wstring(ENWord+"+"));
}

}}  // namespace BASE_NAMESPACE::REGEX

#endif
