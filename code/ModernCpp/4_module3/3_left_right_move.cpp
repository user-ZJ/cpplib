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

    Widget(const Widget& rhs):
        value(rhs.value),
        data(new Point(*rhs.data))
    {
        cout<<"copy ctor"<<endl;

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
     


    ~Widget(){
        cout<<"dtor :"<<data<<endl;
        delete data;
    }


    Widget(Widget&& rhs) noexcept:
        value(rhs.value),
        data(rhs.data)         // 1. 窃取源对象的指针值
    { 
        rhs.data = nullptr;     // 2. 将源对象的值设为有效状态
        cout<<"move ctor"<<endl; 
    }

    
    
    Widget& operator=(Widget&& rhs)	noexcept	
    {	
        if(this==&rhs)
        {
            return *this;
        }

        value=rhs.value;

        delete data;	    // 1. 删除当前值 			
        data = rhs.data;	    // 2. 窃取源对象的值					
        rhs.data = nullptr;	    // 3. 将源对象的值设为有效状态	

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

    //return std::move(w); 不需要！适得其反！
}

void invoke1(Widget&& w)
{

    cout<<"invoke..."<<endl;
}

void invoke2(const Widget& w)
{

    cout<<"invoke..."<<endl;
}



int main()
{


    {
        Widget w1(1,10,20);
        Widget w2(2,100,200);	 
        
        cout<<"-----"<<w1.data<<","<<w2.data<<endl;
        w2 = std::move(w1); 	

        w2.process();
        //w1.process();
        cout<<"-----"<<w1.data<<","<<w2.data<<endl;		          
    }

    cout<<"------"<<endl;
    {
        const Widget w4{3,1000,2000};
        Widget w5{5,1000,2000};
        w5=std::move(w4); 
    }

    cout<<"------"<<endl;
    {
        Widget w3=createWidget();  // 1. 返回值优化 > 2. 移动 > 3. 拷贝
    }


    cout<<"------"<<endl;
    {
        invoke1(createWidget());
    }

    cout<<"------"<<endl;
    {
        invoke2(createWidget());
    }

   
 

    // int data1=100;
    // int data2=std::move(data1);//int data2=data1;

    // Widget *w1=new Widget();
    // Widget* w2=new Widget();
    // w1=std::move(w2); // w1=w2;

    
 }






