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


#include "text-utils.h"
#include <math.h>

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
    SplitStringToVector("", " ", false, &str_vec);
    assert(str_vec.size() == 1);  // If this fails it may just mean
    // that someone changed the
    // semantics of SplitStringToVector in a reasonable way.
    SplitStringToVector("", " ", true, &str_vec);
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
    SplitStringToVector(full, delim.c_str(), omit_empty_strings, &str_vec);
    std::string new_full;
    for (size_t i = 0; i < str_vec.size(); i++) {
      if (omit_empty_strings) assert(str_vec[i] != "");
      new_full.append(str_vec[i]);
      if (i < str_vec.size() -1) new_full.append(delim);
    }
    std::string new_full2;
    JoinVectorToString(str_vec, delim.c_str(), omit_empty_strings, &new_full2);
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


void TestSplitStringOnFirstSpace() {
  std::string a, b;
  SplitStringOnFirstSpace("a b", &a, &b);
  assert(a == "a" && b == "b");
  SplitStringOnFirstSpace("aa bb", &a, &b);
  assert(a == "aa" && b == "bb");
  SplitStringOnFirstSpace("aa", &a, &b);
  assert(a == "aa" && b == "");
  SplitStringOnFirstSpace(" aa \n\t ", &a, &b);
  assert(a == "aa" && b == "");
  SplitStringOnFirstSpace("  \n\t ", &a, &b);
  assert(a == "" && b == "");
  SplitStringOnFirstSpace(" aa   bb \n\t ", &a, &b);
  assert(a == "aa" && b == "bb");
  SplitStringOnFirstSpace(" aa   bb cc ", &a, &b);
  assert(a == "aa" && b == "bb cc");
  SplitStringOnFirstSpace(" aa   bb  cc ", &a, &b);
  assert(a == "aa" && b == "bb  cc");
  SplitStringOnFirstSpace(" aa   bb  cc", &a, &b);
  assert(a == "aa" && b == "bb  cc");
}

void TestIsToken() {
  assert(IsToken("a"));
  assert(IsToken("ab"));
  assert(!IsToken("ab "));
  assert(!IsToken(" ab"));
  assert(!IsToken("a b"));
  assert(IsToken("\231"));  // typical non-ASCII printable character,
                                  // something with an accent.
  assert(!IsToken("\377"));  // character 255, which is a form of space.
  assert(IsToken("a-b,c,d=ef"));
  assert(!IsToken("a\nb"));
  assert(!IsToken("a\tb"));
  assert(!IsToken("ab\t"));
  assert(!IsToken(""));
}

void TestIsLine() {
  assert(IsLine("a"));
  assert(IsLine("a b"));
  assert(!IsLine("a\nb"));
  assert(!IsLine("a b "));
  assert(!IsLine(" a b"));
}


void TestStringsApproxEqual() {
  // we must test the test.
  assert(!StringsApproxEqual("a", "b"));
  assert(!StringsApproxEqual("1", "2"));
  assert(StringsApproxEqual("1.234", "1.235", 2));
  assert(!StringsApproxEqual("1.234", "1.235", 3));
  assert(StringsApproxEqual("x 1.234 y", "x 1.2345 y", 3));
  assert(!StringsApproxEqual("x 1.234 y", "x 1.2345 y", 4));
  assert(StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.49", 1));
  assert(!StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.49", 2));
  assert(StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.411", 2));
  assert(StringsApproxEqual("x 1.0 y", "x 1.0001 y", 3));
  assert(!StringsApproxEqual("x 1.0 y", "x 1.0001 y", 4));
}

void UnitTestConfigLineParse() {
  std::string str;
  {
    ConfigLine cfl;
    str = "a-b xx=yyy foo=bar  baz=123 ba=1:2";
    bool status = cfl.ParseLine(str);
    assert(status && cfl.FirstToken() == "a-b");

    assert(cfl.HasUnusedValues());
    std::string str_value;
    assert(cfl.GetValue("xx", &str_value));
    assert(str_value == "yyy");
    assert(cfl.HasUnusedValues());
    assert(cfl.GetValue("foo", &str_value));
    assert(str_value == "bar");
    assert(cfl.HasUnusedValues());
    assert(!cfl.GetValue("xy", &str_value));
    assert(cfl.GetValue("baz", &str_value));
    assert(str_value == "123");

    std::vector<int32> int_values;
    assert(!cfl.GetValue("xx", &int_values));
    assert(cfl.GetValue("baz", &int_values));
    assert(cfl.HasUnusedValues());
    assert(int_values.size() == 1 && int_values[0] == 123);
    assert(cfl.GetValue("ba", &int_values));
    assert(int_values.size() == 2 && int_values[0] == 1 && int_values[1] == 2);
    assert(!cfl.HasUnusedValues());
  }

  {
    ConfigLine cfl;
    str = "a-b baz=x y z pp = qq ab =cd ac= bd";
    assert(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "a-b baz=x y z pp = qq ab=cd ac=bd";
    assert(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "foo-bar";
    assert(cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "foo-bar a=b c d f=g";
    std::string value;
    assert(cfl.ParseLine(str) && cfl.FirstToken() == "foo-bar" &&
                 cfl.GetValue("a", &value)  && value == "b c d" &&
                 cfl.GetValue("f", &value) && value == "g" &&
                 !cfl.HasUnusedValues());
  }
  {
    ConfigLine cfl;
    str = "zzz a=b baz";
    assert(cfl.ParseLine(str) && cfl.FirstToken() == "zzz" &&
                 cfl.UnusedValues() == "a=b baz");
  }
  {
    ConfigLine cfl;
    str = "xxx a=b baz ";
    assert(cfl.ParseLine(str) && cfl.UnusedValues() == "a=b baz");
  }
  {
    ConfigLine cfl;
    str = "xxx a=b =c";
    assert(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "xxx baz='x y z' pp=qq ab=cd ac=bd";
    assert(cfl.ParseLine(str) && cfl.FirstToken() == "xxx");
    std::string str_value;
    assert(cfl.GetValue("baz", &str_value));
    assert(str_value == "x y z");
    assert(cfl.GetValue("pp", &str_value));
    assert(str_value == "qq");
    assert(cfl.UnusedValues() == "ab=cd ac=bd");
    assert(cfl.GetValue("ab", &str_value));
    assert(str_value == "cd");
    assert(cfl.UnusedValues() == "ac=bd");
    assert(cfl.HasUnusedValues());
    assert(cfl.GetValue("ac", &str_value));
    assert(str_value == "bd");
    assert(!cfl.HasUnusedValues());
  }

  {
    ConfigLine cfl;
    str = "x baz= pp = qq flag=t ";
    assert(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = " x baz= pp=qq flag=t  ";
    assert(cfl.ParseLine(str) && cfl.FirstToken() == "x");

    std::string str_value;
    assert(cfl.GetValue("baz", &str_value));
    assert(str_value == "");
    assert(cfl.GetValue("pp", &str_value));
    assert(str_value == "qq");
    assert(cfl.HasUnusedValues());
    assert(cfl.GetValue("flag", &str_value));
    assert(str_value == "t");
    assert(!cfl.HasUnusedValues());

    bool bool_value = false;
    assert(cfl.GetValue("flag", &bool_value));
    assert(bool_value);
  }

  {
    ConfigLine cfl;
    str = "xx _baz=a -pp=qq";
    assert(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "xx 0baz=a pp=qq";
    assert(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "xx -baz=a pp=qq";
    assert(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "xx _baz'=a pp=qq";
    assert(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = " baz=g";
    assert(cfl.ParseLine(str) && cfl.FirstToken() == "");
    bool flag;
    assert(!cfl.GetValue("baz", &flag));
  }
  {
    ConfigLine cfl;
    str = "xx _baz1=a pp=qq";
    assert(cfl.ParseLine(str));

    std::string str_value;
    assert(cfl.GetValue("_baz1", &str_value));
  }
}

void UnitTestReadConfig() {
  std::string str = "a-b alpha=aa beta=\"b b\"# String test\n"
      "a-b beta2='b c' beta3=bd # \n"
      "a-b gamma=1:2:3:4  # Int Vector test\n"
      " a-b de1ta=f  # Bool + Integer in key Comment test delta=t  \n"
      "a-b _epsilon=-1  # Int Vector test _epsilon=1 \n"
      "a-b zet-_a=0.15   theta=1.1# Float, -, _ test\n"
      "a-b quoted='a b c' # quoted string\n"
      "a-b quoted2=\"d e 'a b=c' f\" # string quoted with double quotes";

  std::istringstream is(str);
  std::vector<std::string> lines;
  ReadConfigLines(is, &lines);
  assert(lines.size() == 8);

  ConfigLine cfl;
  for (size_t i = 0; i < lines.size(); i++) {
    assert(cfl.ParseLine(lines[i]) && cfl.FirstToken() == "a-b");
    if (i == 1) {
        assert(cfl.GetValue("beta2", &str) && str == "b c");
    }
    if (i == 4) {
      assert(cfl.GetValue("_epsilon", &str) && str == "-1");
    }
    if (i == 5) {
      float float_val = 0;
      assert(cfl.GetValue("zet-_a", &float_val) && ApproxEqual(float_val, 0.15));
    }
    if (i == 6) {
      assert(cfl.GetValue("quoted", &str) && str == "a b c");
    }
    if (i == 7) {
      assert(cfl.GetValue("quoted2", &str) && str == "d e 'a b=c' f");
    }
  }
}


int main() {
  TestSplitStringToVector();
  TestSplitStringToIntegers();
  TestSplitStringToFloats();
  TestConvertStringToInteger();
  TestConvertStringToReal<float>();
  TestConvertStringToReal<double>();
  TestTrim();
  TestSplitStringOnFirstSpace();
  TestIsToken();
  TestIsLine();
  TestStringsApproxEqual();
  TestNan<float>();
  TestNan<double>();
  TestInf<float>();
  TestInf<double>();
  UnitTestConfigLineParse();
  UnitTestReadConfig();
  std::cout << "Test OK\n";
}
