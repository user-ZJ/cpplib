
#include <utility>
#include <string>
#include <iostream>
#include <type_traits>
#include <complex>

using namespace std;


namespace MyLib
{
    template <bool, class _Tp = void> 
    struct  enable_if {

    };

    template <class _Tp> struct 
    enable_if<true, _Tp> {typedef _Tp type;};


    //C++ 14提供
    template <bool _Bp, class _Tp = void> 
    using enable_if_t = typename enable_if<_Bp, _Tp>::type;

}



struct MyClass{

};

int main(){

    using Type1=MyLib::enable_if<false>;
    //using T1= Type1::type; // 无类型可取

    using Type2=MyLib::enable_if<true>;
    using T2= Type2::type; // void

    using Type3=MyLib::enable_if<false, MyClass>;
    //using T3= Type3::type; // ? 无类型可取

    using Type4=MyLib::enable_if<true, MyClass>;
    using T4= Type4::type; //?  MyClass

}