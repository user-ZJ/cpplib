#include <utility>
#include <functional>
#include <iostream>
#include <vector>

using namespace std;

template<typename Iter, typename Callable, typename... Args>
void foreach (Iter current, Iter end, Callable op, Args const&... args)
{
  while (current != end) {     
    std::invoke(op,            
                args...,       
                *current);     
    ++current;
  }
}



class MyClass {
  public:

    void print(int data){
        cout<<"MyClass print: "<<data<<endl;
    }

    static void sprint(int data){
      cout<<"MyClass static sprint: "<<data<<endl;
    }
};

void print(int data){
        cout<<"print function pointer: "<<data<<endl;
}

struct Printer {
    void operator() (int data) const {  
      cout  << "Print Functor: "<< data <<endl;
    }
};


template <class _Fp, class _A0>
auto myInvoke(_Fp&& __f, _A0&& __a0)
{
    return __f( std::forward<_A0>(__a0));
}

template <class _Fn, class _T1, class _A0>
auto myInvoke(_Fn&& __f, _T1& __t1, _A0&& __a0) {
    return (__t1.*__f)(std::forward<_A0>(__a0));
}

template <class _Fn, class _T1, class _A0>
auto myInvoke(_Fn&& __f, _T1* __t1, _A0&& __a0) {
    return ((*__t1).*__f)(std::forward<_A0>(__a0));
}



int main()
{
    MyClass mc;

    cout<<"myInvoke---------"<<endl;
    {
      myInvoke(print, 10 );
      myInvoke(Printer{}, 20 );
      myInvoke(&MyClass::print, &mc, 30 );
      myInvoke(&MyClass::print, mc, 30);
      myInvoke(&MyClass::sprint,40);
      myInvoke(
        [](int data){cout<<"lambda:"<<data<<endl;}, 
        50);
      
    }

    
}

