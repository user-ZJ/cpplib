
#include <iostream>
#include <memory>
using namespace std;

template <typename U, typename V> //表示通用版本
struct Printer{
    U x;
    V y;

    void print()
    {
        cout<<x<<","<<y<<endl;
    }
};


template <> //特化版本
struct Printer<int,int>{
    int x;
    int y;

    void print()
    {
        cout<<"["<<x<<","<<y<<"]"<<endl;
    }
};

template <> //特化版本
struct Printer<double,double>{
    double x;
    double y;

    void print()
    {
        cout<<"{"<<x<<","<<y<<"}"<<endl;
    }
};

template <typename V> //偏特化（部分特化）
struct Printer<int,V>{
    int x;
    V y;

    void print()
    {
        cout<<"["<<x<<"] "<<y<<endl;
    }
};

template <typename U, typename V> //指针特化
struct Printer<U*, V*> {
    U* x;
    V* y;

    void print()
    {
        cout<<x<<"->"<<*x<<endl;
        cout<<y<<"->"<<*y<<endl;
    }
};






int main()
{
    Printer<string, string> p1{"C++", "Java"};
    p1.print();

    Printer<int, int> p2{100,200};
    p2.print();

    Printer<double, double> p3{10.2,20.3};
    p3.print();

    Printer<int, string> p4{100,"C++"};
    p4.print();

    auto i1=make_unique<int>(100);
    auto i2=make_unique<int>(200);
    Printer<int*, int*> p5{i1.get(),i2.get()};
    p5.print();

}