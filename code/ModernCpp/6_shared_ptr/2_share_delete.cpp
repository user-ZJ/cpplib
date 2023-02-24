
#include <memory>
#include <iostream>
using namespace std;

class Widget{
    int m_x;
    int m_y;
    int m_z;

public:
    Widget(int x,int y , int z):
        m_x(x), m_y(y),m_z(z)
    {}

    void print(){
        cout<<m_x<<","<<m_y<<","<<m_z<<endl;
    }

    ~Widget()
    {
        cout<<"Widget dtor"<<endl;
    }
};

struct DeleteWithLog { 
    void operator()(Widget* w) { 
      ///  data++;
        delete w;
        cout<<"------ DeleteWithLog delete w"<<endl;
    } 

    //// int data;
};



void deleteFunction(Widget* w) { 
    delete w;
    cout<<"------ deleteFunction delete w"<<endl;
} 

int main()
{

    {
        shared_ptr<Widget> w1(new Widget(1,2,3));
        w1->print();
        cout << sizeof(w1) << endl; 
    }
    cout<<"------ default delete"<<endl;
    

    {
        DeleteWithLog dw;
        shared_ptr<Widget> w2(new Widget(10,20,30),dw);
        w2->print();
        cout << sizeof(w2) << endl; 
    }

    int data1=1,data2=2,data3=3;
    {
        auto lambda = [=](Widget* w) { 
            cout<<data1<<data2<<data3<<endl;
            delete w; 
            cout<<"------ lambda delete w"<<endl;
        };
        shared_ptr<Widget> w3(new Widget(100,200,300), lambda);
        w3->print();
        cout << sizeof(w3) << endl; 
    }

    {
        shared_ptr<Widget> 
            w4(new Widget(1000,2000,3000), deleteFunction);
        w4->print();
        cout << sizeof(w4) << endl; 
    }


}

