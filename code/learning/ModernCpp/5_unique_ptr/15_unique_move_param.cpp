#include <memory>
#include <iostream>

using namespace std;


class Widget{

    
public:	

    int m_data;

    Widget(int data):m_data(data) { cout<<"ctor"<<endl;}
    ~Widget(){ cout<<"dtor"<<endl;}

    Widget(const Widget& rhs){ cout<<"copy ctor"<<endl;}	
    Widget(Widget&& rhs){ cout<<"move ctor"<<endl; }	

    Widget& operator=(Widget&& rhs)	{	
        cout<<"move assignment"<<endl;	
        return *this; 			
    }
	Widget& operator=(const Widget& rhs){
        cout<<"copy assignment"<<endl;
        return *this;
    }	

};





class MyClass {

    unique_ptr<Widget> m_w;
public:
    void setWidget(unique_ptr<Widget> w)
    {
        m_w=std::move(w);
    }
};



//move-only 只支持移动的类型

void process1(unique_ptr<Widget> w) // 传值，抢夺所有权
{
    cout<<"inside process1"<<endl;

    //unique_ptr<Widget> upw=std::move(w);
    cout<<w->m_data<<endl;
}

void process2(unique_ptr<Widget>&& w) //右值传引用，不涉及所有权
{
    cout<<"inside process2"<<endl;

    //unique_ptr<Widget> upw=std::move(w);

    cout<<w->m_data<<endl;
}

int main()
{
 

    {
        MyClass c1;

        unique_ptr<Widget> w{new Widget(10)};
        c1.setWidget(std::move(w));

        
        c1.setWidget(make_unique<Widget>(20));
    }
    cout<<"------"<<endl;
    {
        unique_ptr<Widget> w{new Widget(20)};
        process1(std::move(w));
        cout<<"outside process"<<endl;

    }
    cout<<"------"<<endl;
    {
        unique_ptr<Widget> w{new Widget(30)};
        process2(std::move(w));
        cout<<"outside process"<<endl;

    }
        


}