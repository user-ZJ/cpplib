#include <iostream>
using namespace std;

class Base1{
    int x;
    int y;


public:

    virtual ~Base1()=default;

    Base1(){
        process();//JMP 0x000064
    }
 
    virtual void process() //0x000064
    {
        x++;
    }
};





class Sub1: public Base1{ //继承
public:
     int data;


     void process() override  
    {

    }

    ~Sub1()
    {
        //.....
    }
  
};







void process1(Base1 b)//对象切片，不具多态性
{

    b.process();//非多态调用--编译时绑定 JMP 0x0000040

}
void process2(Base1* b)//保留多态性
{
    //双重身份
    //1. 编译时类型：Base1*
    //2. 运行时类型: 根据参数传递的实际类型决定(runtime)
    b->process();//多态调用 JMP (虚表指针)

}

void process3(Base1& b)//保留多态性
{
    //双重身份
    b.process();//多态调用
}

int main()
{
    Base1 b1;
    Sub1 s1;


    Base1* ps1=new Base1();

    process1(s1);//对象切片
    process1(*ps1);//对象切片

    process2(ps1);
    process2(&b1);
    
    process3(s1);
    process3(*ps1);


    Sub1* ps2=dynamic_cast<Sub1*>(ps1); //多态转型
    ///Sub1* ps2=(Sub1*)ps1; //不安全转型
    ///ps2->data++;

    //ps2->data++;
    ps2->process();

    delete ps1; //1. 调用虚析构函数 2 释放内存


    // Base1* pbarray= new Base1[10];

    Sub1* psarray= new Sub1[10];// 24byte
    Base1* pbarray= psarray; // 16byte
    for(int i=0;i<10;i++)
    {
        pbarray[i].process();//pbarray[i] ==> pbarray+ sizeof(Base1)*i
    }

    delete[] psarray;

}

