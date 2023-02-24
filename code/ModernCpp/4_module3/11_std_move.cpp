#include <iostream>
#include <vector>
using namespace std;




 template<typename _Tp> 
 constexpr typename std::remove_reference<_Tp>::type&& move(_Tp&& __t) noexcept
{ 
    return static_cast<typename std::remove_reference<_Tp>::type&&>(__t); 
}

std::move(T)   ==> static_cast<T&&>(T);

Widget w1;

std::move(w1) ==>  static_cast<Widget&&>(w1);

int&
int&&



// remove_reference<int>::type  -->  int 
// remove_reference<int&>::type  -->  int 
// remove_reference<int&&>::type  -->  int 

// Type Traits 
template <class T> 
struct  remove_reference        
{typedef  T type;};

template <class T> 
struct remove_reference<T&>  
{typedef  T type;};

template <class T> 
struct remove_reference<T&&> 
{typedef  T type;};


