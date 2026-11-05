#include <iostream>
using namespace std;


class Base1{
    int x;
    int y;
    int z;
    int w;
 
public:
    Base1(){}

    Base1(const Base1& r):x(r.x),y(r.y){

    }
    void process1()
    {
        this->x++;
        this->y++;
    }

    // virtual void process()
    // {}

 
};

/*
   void process1(Base1* this)
    {
        this->x++;
        this->y++;
    }
*/

class Base2{
    int u;
    int v;
};

class Sub1: public Base1{
    double data;

};

/*

struct Sub1{
    Base1 base;
    double data;
};
struct Sub1{
    int x;
    int y;
    double data;
};

*/

class Sub2: public Base1, public Base2
{
    double data;
};






int main()
{

    cout<<sizeof(Base1)<<endl;//16
    cout<<sizeof(Base2)<<endl;//8
    cout<<sizeof(Sub1)<<endl;//16+8=24
    cout<<sizeof(Sub2)<<endl;//16+8+8=32

    Base1 b1;
    Sub1 s1;
    b1.process1();// process1(&b1);
    s1.process1();// process1(&s1);

    //s1=b1;
    b1=s1; //对象切割 object slicing

    Base1* pb1=new Base1();
    Sub1* sb1=new Sub1();
    pb1=sb1; //多态指针

    Base1& b2=*sb1;//多态引用






  

}