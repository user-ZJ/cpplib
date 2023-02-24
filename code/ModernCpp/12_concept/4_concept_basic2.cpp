#include<iostream>

using namespace std;

template<typename T>
concept IsPointer = std::is_pointer_v<T>;


template<typename T> 
T add(T a, T b)
{
    return a+b;
}   

template<typename T> 
requires IsPointer<T>
auto add(T a, T b) // 概念要求 是特化版本
{
    cout<<"one time"<<endl;
    return add(*a, *b); 
}



int main()
{
    int x = 100;
    int y = 200;

    int* px=&x;
    int* py=&y;
    cout << add(x, y) << endl; 
    cout << add(px, py) << endl;

    int** ppx=&px;
    int** ppy=&py;
    cout << add(ppx, ppy) << endl;


}