#include <iostream>
#include <any>

using namespace std;



int alloc_times = 0; 
int dealloc_times = 0; 
int allocated = 0; 



void* operator new(size_t size)  { 
  void* p = std::malloc(size); 
  allocated+=size;
  alloc_times++;
  return p; 
} 
 
void operator delete(void* p) noexcept  { 
  dealloc_times++;
  return std::free(p); 
}

void* operator new[](size_t size)   {
  void* p = std::malloc(size); 
  allocated+=size;
  alloc_times++;

  return p; 
} 
void operator delete[](void* p) noexcept  { 
  dealloc_times++;
  return std::free(p); 
}



struct Widget{
    double x{};
    double y{};
    double z{};
    double u{};
    double v{};
    double w{};


    Widget(){}
    Widget(const Widget& w)
    {
        cout<<"copy ctor"<<endl;
    }

    Widget(Widget&& w)
    {
        cout<<"move ctor"<<endl;
    }

    ~Widget(){
        cout<<"dtor"<<endl;
    }
};

int main(){

    cout<<"总分配："<< allocated<<" bytes, 分配次数："<<alloc_times<<" 释放次数："<<dealloc_times<<endl;
  
    {
      Widget w;

      std::any any1=100;
      std::any any2="hello"s;
      std::any any3=w;

         
      cout<<sizeof(std::any)<<endl;
      cout<<sizeof(any1)<<endl;
      cout<<sizeof(any2)<<endl;
      cout<<sizeof(any3)<<endl;
      cout<<sizeof(w)<<endl;
      cout<<(any3.type()== typeid(Widget))<<endl;

     

    }

     cout<<"总分配："<< allocated<<" bytes, 分配次数："<<alloc_times<<" 释放次数："<<dealloc_times<<endl;
  


    


}