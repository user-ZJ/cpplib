// util/text-utils-test.cc

// Copyright 2009-2011     Microsoft Corporation
//                2017     Johns Hopkins University (author: Daniel Povey)
//                2015  Vimal Manohar (Johns Hopkins University)

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


#include "text-util.h"
#include <math.h>

using namespace BASE_NAMESPACE;

char GetRandChar() {
  return static_cast<char>(32 + rand() % 95);  // between ' ' and '~'
}

const char *ws_delim = " \t\n\r";
char GetRandDelim() {
  if (rand() % 2 == 0)
    return static_cast<char>(33 + rand() % 94);  // between '!' and '~';
  else
    return ws_delim[rand() % 4];
}


void TestSplitStringToVector() {
  // srand((unsigned int)time(NULL));
  // didn't compile on cygwin.

  {
    std::vector<std::string> str_vec;
    SplitStringToVector("", " ",  &str_vec,false);
    assert(str_vec.size() == 1);  // If this fails it may just mean
    // that someone changed the
    // semantics of SplitStringToVector in a reasonable way.
    SplitStringToVector("", " ", &str_vec,true);
    assert(str_vec.empty());
  }
  for (int j = 0; j < 100; j++) {
    std::vector<std::string> str_vec;
    int sz = rand() % 73;
    std::string full;
    for (int i = 0; i < sz-1; i++) {
      full.push_back((rand() % 7 == 0)? GetRandDelim() : GetRandChar());
    }
    std::string delim;
    delim.push_back(GetRandDelim());
    bool omit_empty_strings = (rand() %2 == 0)? true : false;
    SplitStringToVector(full, delim.c_str(),  &str_vec,omit_empty_strings);
    std::string new_full;
    for (size_t i = 0; i < str_vec.size(); i++) {
      if (omit_empty_strings) assert(str_vec[i] != "");
      new_full.append(str_vec[i]);
      if (i < str_vec.size() -1) new_full.append(delim);
    }
    std::string new_full2;
    JoinVectorToString(str_vec, delim.c_str(), &new_full2,omit_empty_strings);
    if (omit_empty_strings) {  // sequences of delimiters cannot be matched
      size_t start = full.find_first_not_of(delim),
          end = full.find_last_not_of(delim);
      if (start == std::string::npos) {  // only delimiters
        assert(end == std::string::npos);
      } else {
        std::string full_test;
        char last = '\0';
        for (size_t i = start; i <= end; i++) {
          if (full[i] != last || last != *delim.c_str())
            full_test.push_back(full[i]);
          last = full[i];
        }
        if (!full.empty()) {
          assert(new_full.compare(full_test) == 0);
          assert(new_full2.compare(full_test) == 0);
        }
      }
    } else if (!full.empty()) {
      assert(new_full.compare(full) == 0);
      assert(new_full2.compare(full) == 0);
    }
  }
}

void TestSplitStringToIntegers() {
  {
    std::vector<int32> v;
    assert(SplitStringToIntegers("-1:2:4", ":", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    assert(SplitStringToIntegers("-1:2:4:", ":", false, &v) == false);
    assert(SplitStringToIntegers(":-1::2:4:", ":", true, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    assert(SplitStringToIntegers("-1\n2\t4", " \n\t\r", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    assert(SplitStringToIntegers(" ", " \n\t\r", true, &v) == true
           && v.size() == 0);
    assert(SplitStringToIntegers("", " \n\t\r", false, &v) == true
           && v.size() == 0);
  }

  {
    std::vector<uint32> v;
    assert(SplitStringToIntegers("-1:2:4", ":", false, &v) == false);
    // cannot put negative number in uint32.
  }
}



void TestSplitStringToFloats() {
  {
    std::vector<float> v;
    assert(SplitStringToFloats("-1:2.5:4", ":", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2.5 && v[2] == 4);
    assert(SplitStringToFloats("-1:2.5:4:", ":", false, &v) == false);
    assert(SplitStringToFloats(":-1::2:4:", ":", true, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    assert(SplitStringToFloats("-1\n2.5\t4", " \n\t\r", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2.5 && v[2] == 4);
    assert(SplitStringToFloats(" ", " \n\t\r", true, &v) == true
           && v.size() == 0);
    assert(SplitStringToFloats("", " \n\t\r", false, &v) == true
           && v.size() == 0);
  }

  {
    std::vector<double> v;
    assert(SplitStringToFloats("-1:2:4", ":", false, &v) == true);
  }
}

void TestConvertStringToInteger() {
  int32 i;
  assert(ConvertStringToInteger("12345", &i) && i == 12345);
  assert(ConvertStringToInteger("-12345", &i) && i == -12345);
  char j;
  assert(!ConvertStringToInteger("-12345", &j));  // too big for char.

  assert(ConvertStringToInteger(" -12345 ", &i));  // whitespace accepted

  assert(!ConvertStringToInteger("a ", &i));  // non-integers rejected.

  assert(ConvertStringToInteger("0", &i) && i == 0);

  uint64 k;
  assert(ConvertStringToInteger("12345", &k) && k == 12345);
  assert(!ConvertStringToInteger("-12345", &k));  // unsigned,
                                                        // cannot convert.
}

template<class Real>
void TestConvertStringToReal() {
  Real d;
  assert(ConvertStringToReal("1", &d) && d == 1.0);
  assert(ConvertStringToReal("-1", &d) && d == -1.0);
  assert(ConvertStringToReal("-1", &d) && d == -1.0);
  assert(ConvertStringToReal(" -1 ", &d) && d == -1.0);
  assert(!ConvertStringToReal("-1 x", &d));
  assert(!ConvertStringToReal("-1f", &d));
  assert(ConvertStringToReal("12345.2", &d) && fabs(d-12345.2) < 1.0);
  assert(ConvertStringToReal("1.0e+08", &d) && fabs(d-1.0e+08) < 100.0);

  // it also works for inf or nan.
  assert(ConvertStringToReal("inf", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal(" inf", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal("inf ", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal(" inf ", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal("+inf", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal("-inf", &d) && d < 0 && d - d != 0);
  assert(ConvertStringToReal("Inf", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal("INF", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal("InF", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal("infinity", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal("-infinity", &d) && d < 0 && d - d != 0);
  assert(!ConvertStringToReal("GARBAGE inf", &d));
  assert(!ConvertStringToReal("GARBAGEinf", &d));
  assert(!ConvertStringToReal("infGARBAGE", &d));
  assert(!ConvertStringToReal("inf_GARBAGE", &d));
  assert(!ConvertStringToReal("inf GARBAGE", &d));
  assert(!ConvertStringToReal("GARBAGE infinity", &d));
  assert(!ConvertStringToReal("GARBAGEinfinity", &d));
  assert(!ConvertStringToReal("infinityGARBAGE", &d));
  assert(!ConvertStringToReal("infinity_GARBAGE", &d));
  assert(!ConvertStringToReal("infinity GARBAGE", &d));
  assert(ConvertStringToReal("1.#INF", &d) && d > 0 && d - d != 0);
  assert(ConvertStringToReal("-1.#INF", &d) && d < 0 && d - d != 0);
  assert(ConvertStringToReal("-1.#INF  ", &d) && d < 0 && d - d != 0);
  assert(ConvertStringToReal(" -1.#INF ", &d) && d < 0 && d - d != 0);
  assert(!ConvertStringToReal("GARBAGE 1.#INF", &d));
  assert(!ConvertStringToReal("GARBAGE1.#INF", &d));
  assert(!ConvertStringToReal("2.#INF", &d));
  assert(!ConvertStringToReal("-2.#INF", &d));
  assert(!ConvertStringToReal("1.#INFGARBAGE", &d));
  assert(!ConvertStringToReal("1.#INF_GARBAGE", &d));

  assert(ConvertStringToReal("nan", &d) && d != d);
  assert(ConvertStringToReal("+nan", &d) && d != d);
  assert(ConvertStringToReal("-nan", &d) && d != d);
  assert(ConvertStringToReal("Nan", &d) && d != d);
  assert(ConvertStringToReal("NAN", &d) && d != d);
  assert(ConvertStringToReal("NaN", &d) && d != d);
  assert(ConvertStringToReal(" NaN", &d) && d != d);
  assert(ConvertStringToReal("NaN ", &d) && d != d);
  assert(ConvertStringToReal(" NaN ", &d) && d != d);
  assert(ConvertStringToReal("1.#QNAN", &d) && d != d);
  assert(ConvertStringToReal("-1.#QNAN", &d) && d != d);
  assert(ConvertStringToReal("1.#QNAN  ", &d) && d != d);
  assert(ConvertStringToReal(" 1.#QNAN ", &d) && d != d);
  assert(!ConvertStringToReal("GARBAGE nan", &d));
  assert(!ConvertStringToReal("GARBAGEnan", &d));
  assert(!ConvertStringToReal("nanGARBAGE", &d));
  assert(!ConvertStringToReal("nan_GARBAGE", &d));
  assert(!ConvertStringToReal("nan GARBAGE", &d));
  assert(!ConvertStringToReal("GARBAGE 1.#QNAN", &d));
  assert(!ConvertStringToReal("GARBAGE1.#QNAN", &d));
  assert(!ConvertStringToReal("2.#QNAN", &d));
  assert(!ConvertStringToReal("-2.#QNAN", &d));
  assert(!ConvertStringToReal("-1.#QNAN_GARBAGE", &d));
  assert(!ConvertStringToReal("-1.#QNANGARBAGE", &d));
}

template<class Real>
void TestNan() {
  Real d;
  assert(ConvertStringToReal(std::to_string(sqrt(-1)), &d) && d != d);
}

template<class Real>
void TestInf() {
  Real d;
  assert(ConvertStringToReal(std::to_string(exp(10000)), &d) &&
               d > 0 && d - d != 0);
  assert(ConvertStringToReal(std::to_string(-exp(10000)), &d) &&
               d < 0 && d - d != 0);
}


std::string TrimTmp(std::string s) {
  Trim(&s);
  return s;
}

void TestTrim() {
  assert(TrimTmp(" a ") == "a");
  assert(TrimTmp(" a b  c") == "a b  c");
  assert(TrimTmp("") == "");
  assert(TrimTmp("X\n") == "X");
  assert(TrimTmp("X\n\t") == "X");
  assert(TrimTmp("\n\tX") == "X");
}  // end namespace kaldi




int main() {
  TestSplitStringToVector();
  TestSplitStringToIntegers();
  TestSplitStringToFloats();
  TestConvertStringToInteger();
  TestConvertStringToReal<float>();
  TestConvertStringToReal<double>();
  TestTrim();
  TestNan<float>();
  TestNan<double>();
  TestInf<float>();
  TestInf<double>();
  std::cout << "Test OK\n";
}
