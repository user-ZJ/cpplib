#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <functional>

using namespace std;


struct Widget{
    double x1=1;
    double x2;
    double x3;
    double x4;
    double x5;
    double x6;
    double x7;
    double x8;
    
};

void print(int no, string text)
{
    cout<<"["<<no<<"] "<<text<<endl;
}

using PrintPtr=void (*) (int, string);

// 全局函数
int fn_ptr(int x, int y)
{
    return x + y;
}

// 包含 2 个 int 成员的函数对象类型
struct PlusInts {
    PlusInts(int m_, int n_)
        : m(m_)
        , n(n_)
    {}

    PlusInts(const PlusInts& rhs):m(rhs.m), n(rhs.n)
    {
        cout<<"copy ctor"<<endl;
    }

    ~PlusInts(){
        cout<<"dtor"<<endl;
    }

    int operator()(int x, int y)
    {
        return x + y + m + n;
    }

    int m;
    int n;
};


function<int(int, int)> process()
{
    // 使用 std::function 类型包装函数对象
    PlusInts plus(10,20);

    
    std::function<int(int, int)> g=plus;
    
    
    return g;
}

int main()
{
    {
        PrintPtr p0=print;
        function<void(int,string)> p1=print;
        p1(100,"C++");
        p1(200,"Design Patterns");

        function<void(int,string)> p2=print;
        cout<<(&p1==&p2)<<endl;
    }
    cout<<"----"<<endl;

    {
        PlusInts plus(10,20);
        std::function<int(int, int)> g=plus;
        cout<<sizeof(plus)<<endl;
        cout<<sizeof(g)<<endl;
    }
    cout<<"----"<<endl;

    
    {
        Widget w;

        auto lambda=[=](int d1, int d2)->int { 
            return w.x1+d1+d2;};
        
        function h=lambda; //std::function<int(int, int)>
        cout<<h(3,4)<<endl;
        cout<<lambda(3,4)<<endl;
        cout<<sizeof(lambda)<<endl;
        cout<<sizeof(h)<<endl;

        std::function<int(int, int)> k;
        cout<<sizeof(k)<<endl;

        vector<std::function<int(int, int)>> vfunc;

        ///vector<decltype(lambda)> vlam;

    }
}

