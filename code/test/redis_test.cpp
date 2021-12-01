#include "db/RedisWrapper.h"
#include <iostream>
#include <string>

using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  std::cout << "redis test" << std::endl;
  RedisWrapper redis("10.12.51.12");
  redis.set("k1","testkey");
  redis.set("k2","timeouttest",10);
  std::string value;
  std::cout<<redis.get("k1",&value)<<" "<<value<<std::endl;
  std::cout<<redis.get("k2",&value)<<" "<<value<<std::endl;
  std::cout<<redis.exists("k2")<<std::endl;
  return 0;
}
