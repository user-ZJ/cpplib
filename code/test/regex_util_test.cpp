#include "regex-util.h"
#include "utils/flags.h"
#include "utils/logging.h"
#include <iostream>
#include <regex>
#include <string>

using namespace BASE_NAMESPACE::REGEX;

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
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
  bool ret = match(text,reg_text,&part);
  for(auto &p:part)
    LOG(INFO)<< p;

  // search
  text = "123<xml>value</xml>456<widget>center</widget>hahaha<vertical>window</vertical>the end";
  reg_text = R"(<(.*)>(.*)</(\1)>)";
  std::vector<std::vector<std::string>> searched;
  ret = searchAll(text,reg_text,&searched);
  for(auto &a:searched){
      for(auto &b:a)
        LOG(INFO)<<b;
  }

  // tokenize
  text = "123@qq.vip.com,456@gmail.com,789@163.com,abcd@my.com";
  reg_text = R"(,)";
  auto token = split(text,reg_text);
  for(auto &t:token)
    LOG(INFO)<<t;

  // replaceAll
  text = "001-Neo,002-Lucia";
  reg_text = R"((\d+)-(\w+))";
  auto result = replaceAll(text,reg_text,std::string("$1 name=$2"));
  LOG(INFO)<<result;
  

}