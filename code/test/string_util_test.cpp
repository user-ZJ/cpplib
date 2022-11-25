#include<iostream>
#include<string>
#include "string-util.h"
#include "utils/logging.h"
#include "utils/flags.h"


using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]); 
  google::EnableLogCleaner(3);
  std::string str = "AbC";
  LOG(INFO)<<str<<" "<<toLowercase(str)<<" "<<toUppercase(str);
  assert(toLowercase(str)=="abc");
  assert(toUppercase(str)=="ABC");


}
