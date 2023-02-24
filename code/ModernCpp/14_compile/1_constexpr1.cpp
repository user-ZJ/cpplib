#include <iostream>
#include <vector>
#include <array>
using namespace std;

constexpr int fib(int n)
{
    if(n<=2)
    {
        return 1;
    }
    else 
    {
        return fib(n-1) + fib(n-2);
    }
}







int main()
{
    long value=300000;
    const int d1=100;//编译时常量
    //const int d2=value;//运行时常量

    int myarray1[d1]; //栈数组要求 size编译时确定大小
    int myarray2[value]; //C++ 中是错误, C中是正确（栈变长数组）

    constexpr int d2=100;
    array<int, d1> myArray1;
    array<int, d2*1024> myArray2;

    


    //1,1,2,3,5,8,13,21,34,55
    constexpr int f6= fib(6);
    cout<<f6<<endl;

    constexpr int size2=fib(20);
    array<int, fib(10)> myArray;

    int data=10;
    int f10= fib(data);

    vector<int> vec(fib(data)); //要求运行期，不强制编译期


    // // constexpr int size1=8;
    // // array<int, size1> myArray1;

  

    

}



