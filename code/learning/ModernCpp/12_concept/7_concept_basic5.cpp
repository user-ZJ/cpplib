#include<iostream>

using namespace std;


template<typename T>
concept CanCompute=requires(T x, T y) {
    x + y; // 支持+
    x - y; // 支持 -
    x * y;   //支持 * 
    x / y;
};



// auto compute(CanCompute auto a,CanCompute auto b)
// {
//     return (a+b)*(a-b);
// }   

template<CanCompute T>
T compute(T a,T b)
{

    return (a+b)*(a-b);
    //return a+b;
}   





int main()
{
    int x=200;
    int y=100;

  
    cout << compute(x, y) << endl; 

    string s1="hello ";
    string s2="cpp";

    // cout << compute(s1, s2) << endl; 



}