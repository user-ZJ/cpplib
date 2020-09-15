#include<iostream>
#include<string>
#include "path.h"

int main(){
  std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  std::cout<<"basename:"<<basename(path,true)<<"\n";
  std::cout<<"basename:"<<basename(path,false)<<"\n";
  std::cout<<"basename:"<<suffixname(path)<<"\n";
}
