#include <iostream>
using namespace std;

auto allocated = 0; 

void* operator new(size_t size) {  
  void* p = std::malloc(size); 
  allocated += size; 
  return p; 
} 

void operator delete(void* p) noexcept { 
  return std::free(p); 
} 


void process(const string& s)
{

}


int main() { 

 const char* h="hello";

  //string s="hello"s;
  auto s = std::string{"0123456789,0123456789,"}; 
  string s2=s;
  
  //s.reserve(1000);

  std::cout << "stack size = " << sizeof(s) 
    << ", heap size = " << allocated 
    << ", capacity = " << s.capacity() 
    <<", size= "<<s.size() << '\n'; 
}


struct Long { 
  size_t capacity_{}; 
  size_t size_{}; 
  char* data_{}; //堆
};

struct Short { 
  unsigned char size_{};
  char data_[23]{}; //栈
};

union u_ { 
  Short short_layout_; 
  Long long_layout_; 
};




