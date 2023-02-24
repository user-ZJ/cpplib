#include <iostream>
#include <vector>
using namespace std;


struct Point{
    int x;
    int y;
};

struct Widget {
    int value;
    Point *data;

    Widget(int _value=0, int _x=0, int _y=0):
        value(_value),
        data(new Point{_x,_y})
    {

        cout<<"ctor"<<endl;
    }


    ~Widget(){
        cout<<"dtor :"<<data<<endl;
        delete data;
    }

    Widget(const Widget& rhs):
        value(rhs.value),
        data(new Point(*rhs.data))
    {
        cout<<"copy ctor"<<endl;

    }	

    Widget(Widget&& rhs) noexcept :
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
    }



    void process()
    {
        cout<<value<<": ["<<data->x<<","<<data->y<<"]"<<endl;
        
    }
	  
};



Widget createWidget()
{
    Widget w(1, 10,20);
    return w;
}



int main()
{
    
    {
        Widget w{3,1000,2000};
        w=createWidget(); //移动赋值

    }

   
    cout<<"------"<<endl;
    {
        Widget w1{1,10,20};
        Widget w2{2,100,200};
        w2=std::move(w1); //移动赋值	 
	          
    }

    cout<<"------"<<endl;
    {
        Widget w1{1,10,20};
        Widget w2{2,100,200};
        w2=w1;          //拷贝赋值	 
	          
    }

    cout<<"------"<<endl;
    {
        Widget w1{1,10,20};
        Widget w2 =std::move(w1);//移动构造 
	          
    }
    cout<<"------"<<endl;
    {
        Widget w1=createWidget(); //本应移动构造、但编译器执行了拷贝消除	          
    }

    cout<<"------"<<endl;
    {
        Widget w1{1,10,20};
        w1=w1;	 //自赋值    
    }

    cout<<"------"<<endl;
    {
        Widget w1{1,10,20};
	    w1=std::move(w1);  //移动自赋值       
    }
    

 }






