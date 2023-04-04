#include "utils/flags.h"
#include "utils/logging.h"
#include "utils/regex-util.h"
#include "utils/string-util.h"
#include <iostream>
#include <regex>
#include <string>

using namespace BASE_NAMESPACE::REGEX;
using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::EnableLogCleaner(3);

  std::string text = "This is his face";
  std::string reg_text = R"(\w+)";
  xpressive::sregex_compiler scompiler;
  xpressive::sregex sre = scompiler.compile(reg_text);

  // match
  text = "<html>value</html>";
  reg_text = R"(<(.*)>(.*)</(\1)>)";
  std::vector<std::string> part;
  bool ret = match(text, reg_text, &part);
  for (auto &p : part)
    LOG(INFO) << p;

  // search
  text = "123<xml>value</xml>456<widget>center</widget>hahaha<vertical>window</vertical>the end";
  reg_text = R"(<(.*)>(.*)</(\1)>)";
  std::vector<std::string> matchs;
  std::vector<std::vector<std::string>> searched;
  ret = search(text, reg_text, &matchs);
  ret = searchAll(text, reg_text, &searched);
  for (auto &a : searched) {
    for (auto &b : a)
      LOG(INFO) << b;
  }
  std::wstring wtext = L"您好";
  std::vector<std::wstring> wmatchs;
  ret = search(wtext, to_wstring(ZHWord), &wmatchs);
  LOG(INFO) << "find chinese word:" << ret;
  for (auto s : wmatchs) {
    LOG(INFO) << to_string(s);
  }

  // tokenize
  text = "123@qq.vip.com,456@gmail.com,789@163.com,abcd@my.com";
  reg_text = R"(,)";
  auto token = split(text, reg_text);
  for (auto &t : token)
    LOG(INFO) << t;

  // replaceAll
  text = "001-Neo,002-Lucia";
  reg_text = R"((\d+)-(\w+))";
  auto result = replaceAll(text, reg_text, std::string("$1 name=$2"));
  LOG(INFO) << result;

  std::string regex_str = "\\b(\\w+)\\b";

  // 使用 sregex_compiler 编译正则表达式
  xpressive::sregex_compiler compiler;
  auto regex = compiler.compile(regex_str);

  // 检查编译结果是否成功
  // if (regex.empty()) {
  //   LOG(ERROR) << "Failed to compile regex: " << regex_str << std::endl;
  // }
}
