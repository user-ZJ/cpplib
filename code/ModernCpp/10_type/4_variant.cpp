#include <iostream>
#include <any>
#include <variant>
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



struct WidgetA{
    double x{};
    WidgetA(){

      cout<<"default ctor"<<endl;
    }
    WidgetA(const WidgetA& w)
    {
        cout<<"copy ctor"<<endl;
    }

    WidgetA& operator=(const WidgetA& w)
    {
        cout<<"assignment ="<<endl;
        return *this;
    }

    WidgetA(WidgetA&& w)
    {
        cout<<"move ctor"<<endl;
    }

    ~WidgetA(){
        cout<<"dtor"<<endl;
    }
};

struct WidgetB{
    double x{};
    double y{};
};
struct WidgetC{
    double x{};
    double y{};
    double z{};
    double u{};
    double v{};
    double w{};
};



union WidgetABCU{
    WidgetA a;
    WidgetB b;
    WidgetC c;

    WidgetABCU(){}
    ~WidgetABCU(){}
    
};



int main(){

    using WidgetABC = std::variant<WidgetA, WidgetB,WidgetC>;
    
    cout<<"std::variant : -----------"<<endl;
    {
        WidgetABC w1=WidgetA{};
        WidgetABC w2=w1;
        cout<<sizeof(w1)<<endl;

 

        
    }
    cout<<"union : -----------"<<endl;
    {
        WidgetABCU w3;
        cout<<sizeof(w3)<<endl;
        w3.a= WidgetA();


        //w3.a.~WidgetA();
    }

    cout<<"总分配："<< allocated<<" bytes, 分配次数："<<alloc_times<<" 释放次数："<<dealloc_times<<endl;
  
    


}
