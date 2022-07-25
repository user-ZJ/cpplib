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

template<typename K,typename V>
inline V getFromDic(const std::map<K, V> &dic, const K &key,
                              const V &default_value) {
  auto iter = dic.find(key);
  if (iter != dic.end()) {
    return iter->second;
  } else {
    return default_value;
  }
}

template<typename K,typename V>
inline V getFromDic(const std::unordered_map<K, V> &dic, const K &key,
                              const V &default_value) {
  auto iter = dic.find(key);
  if (iter != dic.end()) {
    return iter->second;
  } else {
    return default_value;
  }
}


};  // namespace BASE_NAMESPACE