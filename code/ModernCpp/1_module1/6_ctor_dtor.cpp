#include <iostream>
#include <string>
#include <memory>
#include <vector>

using namespace std;



class MyClass1
{
public:
    MyClass1()
    {
        cout<<"MyClass1()"<<endl;
    }
    ~MyClass1(){
        cout<<"~MyClass1()"<<endl;
    }
};

class MyClass2
{
public:
    MyClass2()
    {
        cout<<"MyClass2()"<<endl;
    }
    ~MyClass2(){
        cout<<"~MyClass2()"<<endl;
    }
};

class MyClass3
{
public:
    MyClass3()
    {
        cout<<"MyClass3()"<<endl;
    }

    ~MyClass3(){
        cout<<"~MyClass3()"<<endl;
    }
};

class MyClass4
{
public:
    MyClass4()
    {
        cout<<"MyClass4()"<<endl;
    }
    ~MyClass4(){
        cout<<"~MyClass4()"<<endl;
    }
};

class MyClass5  : public MyClass1, public MyClass2
{
    MyClass3 m3;
    MyClass4 m4;
public:
    explicit MyClass5()
    {


        cout<<"MyClass5()"<<endl;
    }

    void process(){
   
    }
    ~MyClass5(){
        cout<<"~MyClass5()"<<endl;
    }
};
int main()
{

    MyClass5* mc5=new MyClass5();
    mc5->process();
    delete mc5;

    
}

