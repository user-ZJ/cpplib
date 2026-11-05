#include <iostream>
using namespace std;

class Base1{

public:

    virtual Base1* clone()=0;

    virtual ~Base1()=default;
};


class Sub1: public Base1{ //继承

public:
    Sub1(){

    }
    Sub1(const Sub1& rhs)
    {
        //....
    }
    
    Base1* clone() override 
    {
        Sub1* pb=new Sub1(*this);

        return pb;

    }

};

class Sub2: public Sub1{ //继承

public:

    Sub2(const Sub2& rhs)
    {
        //....
    }
    
    Base1* clone() override 
    {
        Sub2* pb=new Sub2(*this);

        return pb;

    }

};

int main(){
    return 0;
}

