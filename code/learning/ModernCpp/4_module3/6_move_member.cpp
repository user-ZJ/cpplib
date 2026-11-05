#include <iostream>
#include <vector>
using namespace std;

class MyClass{
public:
    MyClass(){}

    MyClass(const MyClass& rhs)
    { 
        cout<<"MyClass copy ctor"<<endl;
    }
    MyClass(MyClass&& rhs)
    { 
        cout<<"MyClass move ctor"<<endl;
    }

    MyClass& operator=(const MyClass& rhs)
    {
        cout<<"MyClass copy assignment"<<endl;
        return *this;
    }

    MyClass& operator=(MyClass&& rhs)
    {
        cout<<"MyClass move assignment"<<endl;
        return *this;
    }

};

class Point{
    int m_x;
    int m_y;
public:
    Point(int x, int y):m_x(x),m_y(y)
    {
    }
};


class Widget {
    MyClass mc;//对象成员支持移动

    Point* data;
    int value;
    
public:	
    Widget(int x=0, int y=0):data(new Point(x,y))
    {
        cout<<"ctor"<<endl;
    }
    Widget(const Widget& rhs):
        value(rhs.value),
        mc(rhs.mc)	
    {

        if(rhs.data!=nullptr)
        {
            data=new Point(*rhs.data);
        }
        else 
        {
            data=nullptr;
        }
        cout<<"Widget copy ctor"<<endl;

    }	

    Widget(Widget&& rhs): // 1. 窃取源对象的指针值
    	mc(std::move(rhs.mc)),//move ctor
        data(rhs.data),
        value(rhs.value)
    { 

        rhs.data = nullptr;                 // 2. 将源对象的值设为有效状态
        cout<<"Widget move ctor"<<endl; 
    
    }	 

    
    Widget& operator=(Widget&& rhs)		
    {	
        if(this==&rhs)
        {
            return *this;
        }

        value=rhs.value;
        mc=std::move(rhs.mc);//move assignment
        
        delete this->data;	    // 1. 删除当前值 			
        data = rhs.data;	    // 2. 窃取源对象的值					
        rhs.data = nullptr;	    // 3. 将源对象的值设为有效状态	

        cout<<"Widget move assignment"<<endl;	
        return *this; 			
    }

    /*
    Widget& operator=(Widget&& rhs)	noexcept	
    {	
        if(this==&rhs)
        {
            return *this;
        }

        Widget temp(std::move(rhs));//调用移动构造
        swap(value, temp.value);
        swap(mc, temp.mc);
        swap(data, temp.data);

        cout<<"Widget move assignment"<<endl;	
        return *this; 			
    }*/

	Widget& operator=(const Widget& rhs)	
    {
         if(this== &rhs){
            return *this;
        }

        Widget temp(rhs);
        swap(value, temp.value);
        swap(data, temp.data);

        cout<<"Widget copy assignment"<<endl;
        return *this;
    }	


    ~Widget(){
        delete data;
        cout<<"dtor"<<endl;
    }
};




Widget createWidget()
{
    Widget w(10,20);
    return w;
}



int main()
{

    {
        Widget w1;
        w1=createWidget();
    }
    
    cout<<"-------"<<endl;
    {
        Widget w1;
        Widget w2(std::move(w1));
    }

  
}


