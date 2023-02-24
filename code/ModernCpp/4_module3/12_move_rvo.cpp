#include <iostream>
#include <vector>
using namespace std;

struct Point{
    int x;
    int y;
    Point(int _x, int _y):x(_x),y(_y)
    {
    }
};



struct Widget {
    int value;
    Point *data;

    Widget(int _value=0, int _x=0, int _y=0):
        value(_value),
        data(new Point(_x,_y))
    {

        cout<<"ctor"<<endl;
    }

  
    
    
    Widget(const Widget& rhs):
        value(rhs.value),
        data(new Point(*rhs.data))
    {
        cout<<"copy ctor"<<endl;

    }	


    Widget(Widget&& rhs) noexcept:
        value(rhs.value),
        data(rhs.data)         // 1. 窃取源对象的指针值
    { 
        rhs.data = nullptr;     // 2. 将源对象的值设为有效状态
        cout<<"move ctor"<<endl; 
    }


    
	Widget& operator=(const Widget& rhs)	
    {
        if(this== &rhs){
            return *this;
        }

        Widget temp(rhs);
        swap(value, temp.value);
       swap(data, temp.data);

        cout<<"copy assignment"<<endl;
        return *this;
    }	    
     

   


    
    
    Widget& operator=(Widget&& rhs) noexcept
    {	
        if(this==&rhs)
        {
            return *this;
        }

        value=rhs.value;

        delete this->data;	    // 1. 删除当前值 			
        data = rhs.data;	    // 2. 窃取源对象的值					
        rhs.data = nullptr;	    // 3. 将源对象的值设为有效状态	

        cout<<"move assignment"<<endl;	
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
        swap(data, temp.data);

        cout<<"move assignment"<<endl;	
        return *this; 			
    }*/
    


    ~Widget(){
        cout<<"dtor :"<<data<<endl;
        delete data;
    }


    void process()
    {
        cout<<value<<": ["<<data->x<<","<<data->y<<"]"<<endl;
        
    }
	  
};




Widget createWidget_RVO()
{
    Widget w(10,20);
    return w; // 编译器优化选项
}

Widget createWidget_NRVO()
{
    return Widget(10,20); // C++17 标准要求 做拷贝消除--返回值优化
}


int main()
{

    {
        Widget w1=createWidget_RVO();
    }

    cout<<"-------"<<endl;
    
    {
        Widget w1;

        w1=createWidget_NRVO(); //createWidget_NRVO
    }
}


