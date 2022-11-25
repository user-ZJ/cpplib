#include "db/RedisWrapper.h"
#include <iostream>
#include <string>
#include "utils/logging.h"
#include "utils/flags.h"

using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]); 
  LOG(INFO) << "redis test" << std::endl;
  RedisWrapper redis("10.12.51.12");
  redis.set("k1","testkey");
  redis.set("k2","timeouttest",10);
  std::string value;
  LOG(INFO)<<redis.get("k1",&value)<<" "<<value<<std::endl;
  LOG(INFO)<<redis.get("k2",&value)<<" "<<value<<std::endl;
  LOG(INFO)<<redis.exists("k2")<<std::endl;
  
  return 0;
}
