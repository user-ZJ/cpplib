#include <iostream>
#include <vector>
#include <array>
using namespace std;


class Point{
    int m_x;
    int m_y;
public:
    Point(int x, int y):m_x(x),m_y(y)
    {
    }
};

class Widget {
    Point *data;
    int value;

public:	
    Widget(int x=0, int y=0):data(new Point(x,y))
    {
    }
    Widget(const Widget& rhs) noexcept:value(rhs.value)	
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

  
    Widget(Widget&& rhs) noexcept : 
        data(rhs.data),// 1. 窃取源对象的指针值
    	value(rhs.value) 
    { 
        rhs.data = nullptr;   // 2. 将源对象的值设为有效状态
        cout<<"move ctor"<<endl; 
    }	    

    Widget& operator=(Widget&& rhs)	noexcept
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

	Widget& operator=(const Widget& rhs)noexcept	
    {
        if(this== &rhs){
            return *this;
        }

        if(rhs.data!=nullptr){
            if(data!=nullptr){
                *data=*rhs.data;
            }
            else{
                data=new Point(*rhs.data);
            }
        }
        else
        {
            delete data;
            data=nullptr;
        }

        cout<<"copy assignment"<<endl;
        return *this;
    }	


    ~Widget(){
        delete data;
    }
};


vector<Widget> getVec()
{
  vector<Widget> vec(1000);

  return vec;
}



array<Widget,20> getArr()
{
  array<Widget,20> arr;

  return arr;
}



int main()
{
    
    // vector<Widget>  vm;
    // vm=getVec();

    
    // array<Widget,20> am;
    // am=getArr(); //元素移动
    
    
    vector<Widget> vw;
    for(int i=0;i<30;i++)
    {
        vw.emplace_back(i*10,i*20); //元素移动
        cout<<vw.size()<<" / "<<vw.capacity()<<endl;
    }

}





