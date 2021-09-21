#include<iostream>
#include<string>
#include "path.h"
#include "utils/logging.h"


using namespace BASE_NAMESPACE;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]); 
  std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  LOG(INFO)<<"basename:"<<basename(path,true)<<"\n";
  LOG(INFO)<<"basename:"<<basename(path,false)<<"\n";
  LOG(INFO)<<"basename:"<<suffixname(path)<<"\n";
}
