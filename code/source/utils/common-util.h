/*
 * @Author: zack 
 * @Date: 2022-06-07 09:07:53 
 * @Last Modified by:   zack 
 * @Last Modified time: 2022-06-07 09:07:53 
 */
#pragma once
#include <map>
#include <string>
#include <unordered_map>

namespace BASE_NAMESPACE {

inline std::string getFromDic(const std::map<std::string, std::string> &dic, const std::string &key,
                              const std::string &default_value = "") {
  auto iter = dic.find(key);
  if (iter != dic.end()) {
    return iter->second;
  } else {
    return default_value;
  }
}

inline std::string getFromDic(const std::unordered_map<std::string, std::string> &dic, const std::string &key,
                              const std::string &default_value = "") {
  auto iter = dic.find(key);
  if (iter != dic.end()) {
    return iter->second;
  } else {
    return default_value;
  }
}

inline std::wstring getFromDic(const std::map<std::wstring, std::wstring> &dic, const std::wstring &key,
                               const std::wstring &default_value = L"") {
  auto iter = dic.find(key);
  if (iter != dic.end()) {
    return iter->second;
  } else {
    return default_value;
  }
}

inline std::wstring getFromDic(const std::unordered_map<std::wstring, std::wstring> &dic, const std::wstring &key,
                               const std::wstring &default_value = L"") {
  auto iter = dic.find(key);
  if (iter != dic.end()) {
    return iter->second;
  } else {
    return default_value;
  }
}

};  // namespace BASE_NAMESPACE