#include <iostream>
#include <string>

using namespace std;

struct Base{

    Base(){
        cout<<"Base.process()"<<endl;
    }

    Base(int data){
        cout<<"Base.process(int)"<<endl;
    }

    Base(string text){
        cout<<"Base.process(string)"<<endl;
    }

    void process()
    {
        cout<<"process()"<<endl;
    }

    void process(double data)
    {
        cout<<"process(double data)"<<endl;
    }

};

struct Sub:  Base{
    using Base::Base;
    using Base::process;

    //  Sub():Base(){}
    // Sub(int data):Base(data){}
    //  Sub(string text):Base(text){}


    void process(int data)
    {
        cout<<"process(int data)"<<endl;
    }
};

int main()
{
    Sub s1;
    Sub s2(100);
    Sub s3("hello");

     s3.process(100);
     //s3.Base::process();
     s3.process();
    s3.process(10.23);
}