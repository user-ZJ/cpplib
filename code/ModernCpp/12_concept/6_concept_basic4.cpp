#include<iostream>
#include <memory>
using namespace std;

template<typename T>
 concept IsPointer = std::is_pointer_v<T>;

// template<typename T>
// concept IsPointer = requires(T p) { 
//     *p; 
// };

// template<typename T>
// concept IsPointer = requires(T p) {
//     *p; // 操作符有效
//     {p < p} -> std::convertible_to<bool>; // 产生bool值
//     p == nullptr; // 可以和nullptr比较（迭代器就不行)
// };



auto add(auto a, auto b)
{
    return a+b;
}   

auto add(IsPointer auto& a, IsPointer auto& b) 
{
    return add(*a, *b); 
}

// template<typename T>
// auto add(T& a, T& b)
// {
//     if constexpr(IsPointer<T>)
//     {
//         return (*a)+(*b);
//     }
//     else 
//     {
//         return a+b;
//     }
// }

int main()
{
    auto upx=make_unique<int>(100);
    auto upy=make_unique<int>(200);

  
    // cout << add(upx, upy) << endl; 

    int d1=1000;
    int d2=2000;
    cout<< add(d1,d2)<<endl;




}