// util/text-utils.h

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <errno.h>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <limits>
#include <assert.h>
#include <sstream>
#include <iostream>
#include "base-type.h"


namespace BASE_NAMESPACE{

template <class T>
class NumberIstream{
 public:
  explicit NumberIstream(std::istream &i) : in_(i) {}

  NumberIstream & operator >> (T &x) {
    if (!in_.good()) return *this;
    in_ >> x;
    if (!in_.fail() && RemainderIsOnlySpaces()) return *this;
    return ParseOnFail(&x);
  }

 private:
  std::istream &in_;

  bool RemainderIsOnlySpaces() {
    if (in_.tellg() != std::istream::pos_type(-1)) {
      std::string rem;
      in_ >> rem;

      if (rem.find_first_not_of(' ') != std::string::npos) {
        // there is not only spaces
        return false;
      }   
    }   

    in_.clear();
    return true;
  }

  NumberIstream & ParseOnFail(T *x) {
    std::string str;
    in_.clear();
    in_.seekg(0);
    // If the stream is broken even before trying
    // to read from it or if there are many tokens,
    // it's pointless to try.
    if (!(in_ >> str) || !RemainderIsOnlySpaces()) {
      in_.setstate(std::ios_base::failbit);
      return *this;
    }

    std::map<std::string, T> inf_nan_map;
    // we'll keep just uppercase values.
    inf_nan_map["INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["+INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-INF"] = - std::numeric_limits<T>::infinity();
    inf_nan_map["INFINITY"] = std::numeric_limits<T>::infinity();
    inf_nan_map["+INFINITY"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-INFINITY"] = - std::numeric_limits<T>::infinity();
    inf_nan_map["NAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["+NAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["-NAN"] = - std::numeric_limits<T>::quiet_NaN();
    // MSVC
    inf_nan_map["1.#INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-1.#INF"] = - std::numeric_limits<T>::infinity();
    inf_nan_map["1.#QNAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["-1.#QNAN"] = - std::numeric_limits<T>::quiet_NaN();

    std::transform(str.begin(), str.end(), str.begin(), ::toupper);

    if (inf_nan_map.find(str) != inf_nan_map.end()) {
      *x = inf_nan_map[str];
    } else {
      in_.setstate(std::ios_base::failbit);
    }

    return *this;
  }
};

/// 将字符串转化为float 或 double
/// 如果字符串不能转换，则返回false
/// 注意，此函数可以正确转化inf和Nan.
template <typename T>
bool ConvertStringToReal(const std::string &str,
                         T *out){
  std::istringstream iss(str);

  NumberIstream<T> i(iss);

  i >> *out;

  if (iss.fail()) {
    // Number conversion failed.
    return false;
  }

  return true;
}

template
bool ConvertStringToReal(const std::string &str,
                         float *out);
template
bool ConvertStringToReal(const std::string &str,
                         double *out);

/// 使用delim将字符串分割成子字符串并保存在vector中
/// omit_empty_strings ：是否忽略空字符串
/// SplitStringToVector(std::string("aqwaioazx"),"a",out)
/// 结果为{"qw","io","zx"}
void SplitStringToVector(const std::string &full, const char *delim,
                         std::vector<std::string> *out,bool omit_empty_strings=true){
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

/// 使用delim串将vec_in中字符串连接起来
/// omit_empty_strings :是否忽略空白字符串
/// vec_in = {"qa","ws","ed"}  delim="_"
/// result: qa_ws_ed
void JoinVectorToString(const std::vector<std::string> &vec_in,
                        const char *delim,std::string *str_out,bool omit_empty_strings=true){
  std::string tmp_str;
  for (size_t i = 0; i < vec_in.size(); i++) {
    if (!omit_empty_strings || !vec_in[i].empty()) {
      tmp_str.append(vec_in[i]);
      if (i < vec_in.size() - 1)
        if (!omit_empty_strings || !vec_in[i+1].empty())
          tmp_str.append(delim);
    }
  }
  str_out->swap(tmp_str);
}

/**
  \brief 将字符串分割成int数组 (e.g. 1:2:3) 

  \param [in]  delim  分隔符，可以是任何字符串
  \param [in] omit_empty_strings 如果delim为空格字符串，设置为true，其他时候为false
  \param [out] out   The output list of integers.
*/
template<class I>
bool SplitStringToIntegers(const std::string &full,
                           const char *delim,
                           bool omit_empty_strings,  
                           std::vector<I> *out) {
  assert(out != NULL);
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, &split, omit_empty_strings);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    const char *this_str = split[i].c_str();
    char *end = NULL;
    int64 j = 0;
    j = strtoll(this_str, &end, 10);
    if (end == this_str || *end != '\0') {
      out->clear();
      return false;
    } else {
      I jI = static_cast<I>(j);
      if (static_cast<int64>(jI) != j) {
        // output type cannot fit this integer.
        out->clear();
        return false;
      }
      (*out)[i] = jI;
    }
  }
  return true;
}

// This is defined for F = float and double.
template<class F>
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out){
  assert(out != NULL);
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, &split,omit_empty_strings);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    F f = 0;
    if (!ConvertStringToReal(split[i], &f))
      return false;
    (*out)[i] = f;
  }
  return true;
}

// Instantiate the template above for float and double.
template
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,
                         std::vector<float> *out);
template
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,
                         std::vector<double> *out);


/// 将字符串转换为int，如果转换错误返回false.
template<class Int>
bool ConvertStringToInteger(const std::string &str,
                            Int *out) {
  const char *this_str = str.c_str();
  char *end = NULL;
  errno = 0;
  int64 i = strtoll(this_str, &end, 10);
  if (end != this_str)
    while (isspace(*end)) end++;
  if (end == this_str || *end != '\0' || errno != 0)
    return false;
  Int iInt = static_cast<Int>(i);
  if (static_cast<int64>(iInt) != i ||
      (i < 0 && !std::numeric_limits<Int>::is_signed)) {
    return false;
  }
  *out = iInt;
  return true;
}


/// 从字符串中删除开头和结尾的空格
void Trim(std::string *str){
  const char *white_chars = " \t\n\r\f\v";

  std::string::size_type pos = str->find_last_not_of(white_chars);
  if (pos != std::string::npos)  {
    str->erase(pos + 1);
    pos = str->find_first_not_of(white_chars);
    if (pos != std::string::npos) str->erase(0, pos);
  } else {
    str->erase(str->begin(), str->end());
  }
}

/// Returns true if 'name'  is a nonempty string beginning with A-Za-z_, and containing only
/// '-', '_', '.', A-Z, a-z, or 0-9.
bool IsValidName(const std::string &name){
  if (name.size() == 0) return false;
  for (size_t i = 0; i < name.size(); i++) {
    if (i == 0 && !isalpha(name[i]) && name[i] != '_')
      return false;
    if (!isalnum(name[i]) && name[i] != '_' && name[i] != '-' && name[i] != '.')
      return false;
  }
  return true;
}

};