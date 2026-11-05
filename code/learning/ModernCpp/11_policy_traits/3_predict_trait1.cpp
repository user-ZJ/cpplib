#include <iostream>

using namespace std;

template<bool val>
struct BoolConstant {
  using Type = BoolConstant<val>;
  static constexpr bool value = val;
};


using TrueType  = BoolConstant<true>;
using FalseType = BoolConstant<false>;

template<typename T1, typename T2>
struct IsSameT : FalseType
{
};

template<typename T>
struct IsSameT<T, T> : TrueType
{
};




template<typename T>
void processImpl(T, TrueType)
{
  std::cout << "processImpl (T,true)"<<endl;
}

template<typename T>
void processImpl(T, FalseType)
{
  std::cout << "processImpl (T,false) "<<endl;
}

template<typename T>
void process(T t)
{
  processImpl(t, IsSameT<T,int>{});  
}


template<typename T>
struct IsPointerT : std::false_type {    
};

template<typename T>
struct IsPointerT<T*> : std::true_type { 
  using BaseT = T;  
};


template<typename T>
void invokeImpl(T, std::false_type)
{
  std::cout << "invokeImpl (T,false) "<<endl;
}

template<typename T>
void invokeImpl(T , std::true_type)
{
  std::cout << "invokeImpl (T,true) "<<endl;
}


template<typename T>
void invoke(T t)
{
  invokeImpl(t, std::is_pointer<T>{});  
}




int main()
{
  process(42);   
  process(7.7);  

  int data=100;
  int* pdata=&data;
  invoke(data); //std::cout << "invokeImpl (T,false) "<<endl;
  invoke(pdata);

}
