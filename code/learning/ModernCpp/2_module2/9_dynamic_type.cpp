#include <iostream>
using namespace std;

class Base1{
    int x;
    int y;
public:

    virtual ~Base1()=default;

    virtual void process()
    {

    }
};


class Sub1: public Base1{ //继承

public:
    int value;
    
    void process() override
    {
        Base1::process();
    }

    void func()
    {

    }

};



void process(Base1* pb)//多态基类
{
    pb->process();//多态辨析 dynamic dispatch
    
    Sub1* ps=dynamic_cast<Sub1*>(pb); //多态转型

    //Sub1* p2=(Sub1*)pb; C-转型

    if(ps!=nullptr)
    {
        cout<<"pb是一个Sub1*"<<endl;

        ps->func();// 非多态辨析
    }
    else{
        cout<<"pb不是一个Sub1*"<<endl;
    }
    
}


int main()
{

    Base1* p1=new Base1();
    Base1* p2=new Sub1();
    
    process(p1);
    process(p2);


    Base1& b1=*p2;
    Sub1& s2=dynamic_cast<Sub1&>(b1);
    s2.value++;




  
    Base1* p3=new Sub1();



    //全局唯一、只读
    const type_info& t1=typeid(*p1);
    const type_info& t2=typeid(*p2);
    const type_info& t3=typeid(*p3);

    cout<<t1.name()<<endl;
    if(&t1==&t2)
    {
        cout<<"t1==t2"<<endl;
    }

    if(&t2==&t3)
    {
        cout<<"t2==t3"<<endl;
    }
    
    
    delete p1; 
    delete p2; 



}

