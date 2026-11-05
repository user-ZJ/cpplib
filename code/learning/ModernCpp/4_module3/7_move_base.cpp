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


class Widget :public MyClass{
    Point* data;
    int value;
    
public:	
    Widget(int x=0, int y=0):data(new Point(x,y))
    {
        cout<<"ctor"<<endl;
    }


    Widget(const Widget& rhs):value(rhs.value),MyClass(rhs)	
    {

        if(rhs.data!=nullptr)
        {
            data=new Point(*rhs.data);
        }
        else 
        {
            data=nullptr;
        }
        cout<<"copy ctor"<<endl;

    }	

    Widget(Widget&& rhs) noexcept: // 1. 窃取源对象的指针值
    	MyClass(std::move(rhs)),//调用父类move ctor
        data(rhs.data),
        value(rhs.value)
    { 

        rhs.data = nullptr;                 // 2. 将源对象的值设为有效状态
        cout<<"Widget move ctor"<<endl; 
    
    }	    
    Widget& operator=(Widget&& rhs) noexcept		
    {	
        if(this==&rhs)
        {
            return *this;
        }


        MyClass::operator=(std::move(rhs));// 调用父类move assignment
        value=rhs.value;
        

        delete this->data;	    // 1. 删除当前值 			
        this->data = rhs.data;	    // 2. 窃取源对象的值					
        rhs.data = nullptr;	    // 3. 将源对象的值设为有效状态
       	

        cout<<"Widget move assignment"<<endl;	
        return *this; 			
    }

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








/*
int&
int&&

std::remove_reference<_Tp>::type===> int
std::remove_reference<_Tp>::type&& ===> int&&

 template<typename _Tp> 
 constexpr typename std::remove_reference<_Tp>::type&& move(_Tp&& __t) noexcept
{ 
    return static_cast<typename std::remove_reference<_Tp>::type&&>(__t); 
}

std::move(T)   ==> static_cast<T&&>(T);

*/
