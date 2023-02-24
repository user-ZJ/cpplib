#include <iostream>
#include <memory>
using namespace std;

class MyClass{

public:
    MyClass(){ cout<<"MyClass()"<<endl;}
    ~MyClass(){ cout<<"~MyClass()"<<endl;}
    
 
private:
  double x,y,z;
};



 

int alloc_times = 0; 
int dealloc_times = 0; 
int allocated = 0; 


void* operator new(size_t size)  { 
  void* p = std::malloc(size); 
  cout << "new " << size << " byte(s)"<<endl; 

  allocated+=size;
  alloc_times++;
  return p; 
} 
 
void operator delete(void* p) noexcept  { 
  cout << "deleted memory"<<endl; 

  dealloc_times++;
  return std::free(p); 
}

void heap_monitor(){
  cout<<"分配尺寸:"<<allocated<<" 分配次数:"<<alloc_times<<" 释放次数:"<<dealloc_times<<endl;
}
 
int main(){

    weak_ptr<MyClass> w;
    heap_monitor();
    {
        //shared_ptr<MyClass> s{new MyClass()};
        shared_ptr s = std::make_shared<MyClass>();
        //w=s; 
       
   }   //1. s析构、释放原始对象、但不释放控制块
    cout<<"block end--------\n\n";
    heap_monitor();
  
    // cout<<"after reset-------\n\n";
    // w.reset(); //2. w析构、释放控制块
    // heap_monitor();
    
    
}

