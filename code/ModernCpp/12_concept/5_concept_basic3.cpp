#include<iostream>

using namespace std;

template<typename T>
concept IsPointer = std::is_pointer_v<T>;


// template<typename T> 
// T add(T a, T b)
// {
//     return a+b;
// }   

// template<IsPointer T>
// auto add( T a,  T b) 
// {
//     return add(*a, *b); 
// }


auto add(auto a, auto b)
{
    return a+b;
}   

auto add(IsPointer auto a, IsPointer auto b)
{
    return *a+*b;
}   





int main()
{
    int x = 100;
    int y = 200;

    auto px=&x;
    auto py=&y;
    cout << add(x, y) << endl; 
    cout << add(px, py) << endl;

    
    IsPointer auto pdata=&y;
    floating_point auto data=10.34;

}