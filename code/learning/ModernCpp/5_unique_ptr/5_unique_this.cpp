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

    unique_ptr<Widget> getWidget() {

        //...

        
         return unique_ptr<Widget>( this); //delete this 错误! 
        

         //return unique_ptr<Widget>( new Widget(*this)); //正确!
    }

    void process(){
        unique_ptr<Widget> u1(this); 
        u1->print();

        //delete this;
    }

    ~Widget()
    {
        cout<<"Widget dtor"<<endl;
    }
};

/*
unique_ptr<Widget> getWidget(Widget* this)
{
    return unique_ptr<Widget>(this);
}
*/




int main()
{

  
    {
        Widget* p=new Widget(1,2,3);
        unique_ptr<Widget> w1{p};

        //w1->process();
    }

    {
        //unique_ptr<Widget> w1{new Widget(1,2,3)};

        Widget* w1{new Widget(1,2,3)};

        unique_ptr<Widget> w2=w1->getWidget();// getWidget( w1.get());

        delete w1;

    }

 
}





