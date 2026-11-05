#include <iostream>
#include <vector>
using namespace std;




struct Widget {

    Widget()
    {
        cout<<"ctor"<<endl;
    }

    Widget(const Widget& rhs)
    {
        cout<<"copy ctor"<<endl;

    }	

    Widget& operator=(const Widget& rhs)	
    {

        cout<<"copy assignment"<<endl;
        return *this;
    }	


    Widget(Widget&& rhs)
    { 
        cout<<"move ctor"<<endl; 
    }
    
    Widget& operator=(Widget&& rhs)		
    {	
        cout<<"move assignment"<<endl;	
        return *this; 			
    }


    ~Widget(){
        cout<<"dtor"<<endl;

    }

	  
};



struct MyClass{

    
    void setValue(const Widget& w)
    {
        //....
        //....
        m_w=w; //左值 copy 赋值
    }

    void setValue(Widget&& w)
    {
        //....
        //....
        m_w=std::move(w); //右值 move 赋值
    }


    // template<typename T>
    // void setValue(T&& t)
    // {
    //     //....
    //     //....
    //     m_w=std::forward<T>(t);
    // }


    //1. 如果传递左值，先copy构造，后 move赋值
    //2. 如果传递右值，copy被消除，直接move赋值
    // void setValue(Widget w)
    // {
    //     m_w=std::move(w);
        
    // }

    Widget m_w;

};


int main()
{

    
    MyClass c;
    cout<<"block 0-----"<<endl;
    {
        Widget w;
        c.setValue(w);
    }
    cout<<"block 1-----"<<endl;
    {
        c.setValue( Widget{} );	 
    }
    cout<<"block 2-----"<<endl;
   
    
}





