#include <boost/functional/hash.hpp>
#include <iostream>
#include <string>

using namespace std;

string reverseWords(string s) {
  std::string result;
  int i = s.size() - 1, j = s.size() - 1;
  while (i >= 0 && j >= 0) {
    while (j >= 0 && s[j] == ' ') {
      --j;
      --i;
    }
    while (i >= 0 && s[i] != ' ') {
      --i;
    }
    int start = i < 0 ? 0 : i + 1;
    if (j > i) {
      cout << start << " " << i << " " << j<<s.substr(start, j - i)<<"\n";
      result += s.substr(start, j - i) + " ";
      j = i;
    }
  }
  if (result.back() == ' ') result.pop_back();
  return result;
}

int main() {
  string str = " asdasd df f";
  reverseWords(str);
  return 0;
}