#include<iostream>
#include<string>
#include<vector>
#include "utils/ctensor.h"
#include "utils/logging.h"
#include "utils/flags.h"
#include "utils/string-util.h"


using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]); 
  CTensor<float,int> t1({3,5,7,9});
  std::cout<<"shapes:"<<v2s(t1.shapes())<<std::endl;
  std::cout<<"strides:"<<v2s(t1.strides())<<std::endl;
  std::cout<<"size:"<<t1.size()<<std::endl;
  std::cout<<"byte size:"<<t1.byteSize()<<std::endl;
  std::cout<<t1.at({0,0,0,1})<<std::endl;
  t1.at({0,0,0,1}) += 1.0;
  std::cout<<t1.at({0,0,0,1})<<std::endl;

  std::cout<<"####################"<<std::endl;
  std::vector<int> shape = {3,5,7,9};
  CTensor<float,int> t2(shape);
  std::cout<<"shapes:"<<v2s(t2.shapes())<<std::endl;
  std::cout<<"strides:"<<v2s(t2.strides())<<std::endl;
  std::cout<<"size:"<<t2.size()<<std::endl;
  std::cout<<"byte size:"<<t2.byteSize()<<std::endl;
  std::cout<<"ctensor at:"<<t2.at({1,2,3,4})<<std::endl;
  t2.at({0,0,0,0}) = 1.6;
  t2.at({0,0,0,2}) = 2.999999999999999999999999999999999;
  t2.at({0,0,0,3}) = 2.4;


}
