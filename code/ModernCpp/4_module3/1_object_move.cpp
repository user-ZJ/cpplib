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
}


vector<Widget> getData()
{
      vector<Widget> vec(10);
      return vec;
}




int main()
{
    {
        Widget w1{1,10,20};
        Widget w2{2,100,200};
        w1=w2;

    }

    cout<<"-------"<<endl;
    {
        Widget w{3,1000,2000};
        w=createWidget(); //移动赋值

    }


    
    cout<<"-------"<<endl;
    {
        vector<Widget> v;

        for(int i=0;i<20;i++)
        {   
            cout<<"---"<<i<<"---:"<<v.capacity() <<"," <<v.size()<<endl;
            v.emplace_back(i,i*10,i*20);
            
        }
    }


    cout<<"-------"<<endl;
    {
        vector<Widget> v;
        v=getData();
    }
 }






