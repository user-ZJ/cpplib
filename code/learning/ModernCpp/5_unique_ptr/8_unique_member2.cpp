#include <memory>
#include <iostream>

using namespace std;

struct Widget {
    int m_number;

    Widget(int number=0):m_number(number)
    {
        cout<<"Widget()"<<endl;
    }
    Widget(const Widget& w):m_number(w.m_number)
    {
        cout<<"Widget(const Widget& w)"<<endl;
    }

     Widget(Widget&& w):m_number(w.m_number)
    {
        cout<<"Widget(Widget&& w)"<<endl;
    }
    ~Widget()
    {
        cout<<"~Widget()"<<endl;
    }
};




struct MyClass {

    //Widget* w;

    std::unique_ptr<Widget> m_p;

    

    MyClass(int data): m_p{make_unique<Widget>(data)}
    {

    }

    
    MyClass(const MyClass& rhs):m_p(make_unique<Widget>(*rhs.m_p))
    {
        //m_p= unique_ptr<Widget>{ new Widget(*rhs.m_p) };
    }

    MyClass& operator=(const MyClass& rhs)
    {
        if(this==&rhs) return *this;

        MyClass temp(rhs);
        m_p.swap(temp.m_p);
        return *this;
    }

    MyClass(MyClass&& rhs) noexcept = default;
    MyClass& operator=(MyClass&& rhs) noexcept = default;
    ~MyClass() = default;
};



int main()
{

    MyClass a(100);

    MyClass b{a}; 

    MyClass c=std::move(b);

    cout<<std::boolalpha;
    cout<<(a.m_p==nullptr)<<endl;
    cout<<(b.m_p==nullptr)<<endl;
    cout<<(c.m_p==nullptr)<<endl;
   
    



}