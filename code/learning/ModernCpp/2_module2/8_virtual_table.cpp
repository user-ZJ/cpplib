#include <iostream>

using namespace std;


class Base{
public:
    long long d1;

    virtual void M1(){

    }

    virtual ~Base()=default;
};



class Sub: public Base{
public:
    long long d2;

     void M1() override{

    }
  
    virtual void M2(){

    }
};

int main(){
    Base b;
    b.d1=10;

    Sub s;
    s.d1=100;
    s.d2=200;
 
    cout<<sizeof(b)<<endl;//8+8=16
    cout<<sizeof(s)<<endl;//16+8=24

    long long* p1=(long long*)&b;
    long long* pvt1=p1;
    cout<<*p1<<",";
    p1++;
    cout<<*p1<<endl;

    long long* p2=(long long*)&s;
    long long* pvt2=p2;
    cout<<*p2<<",";
    p2++;
    cout<<*p2<<",";
    p2++;
    cout<<*p2<<endl;

    

    cout<<*pvt1<<":"<< *((long long*)(*pvt1))<<endl;
    cout<<*pvt2<<":"<< *((long long*)(*pvt2))<<endl;
  


}

