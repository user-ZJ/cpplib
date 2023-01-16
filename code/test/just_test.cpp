#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

class Target {
 public:
  virtual void Request() {
    std::cout << "普通请求" << std::endl;
  }
  virtual ~Target() = default;
};

class Adaptee {
 public:
  void SpecificRequest() {
    std::cout << "特殊请求" << std::endl;
  }
};

class Adapter : public Target {
 public:
  Adapter(Adaptee *ad){
    adaptee = ad;
  }
  virtual void Request() override{
    adaptee->SpecificRequest();
  }

 private:
  Adaptee *adaptee;
};

int main(int argc, char *argv[]) {
  Adaptee *adaptee = new Adaptee(); 
  Target *target = new Adapter(adaptee);
  target->Request();
  return 0;
}
