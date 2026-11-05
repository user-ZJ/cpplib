#include <iostream>
#include <memory>
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
    MyClass mc;
    Point* data;
    int value;
    
public:	
    Widget(int _value=0, int _x=0, int _y=0):
        value(_value),
        data(new Point(_x,_y))
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

    Widget(Widget&& rhs) noexcept: 
        mc(std::move(rhs.mc)),
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

	Widget& operator=(const Widget& rhs)	
    {
        if(this== &rhs){
            return *this;
        }

        value=rhs.value;
        mc=rhs.mc;

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
        cout<<"dtor"<<endl;
    }
};




int main()
{
  
  cout<<boolalpha<<endl;


  cout << is_copy_constructible<MyClass>::value << '\n';      // true
  cout << is_copy_constructible_v<MyClass> << '\n';      // true 
  cout<< is_move_constructible_v<MyClass> <<"\n";        //true
  cout<< is_copy_assignable_v<MyClass> <<'\n';          //true
  cout<< is_move_assignable_v<MyClass> <<'\n';          //true

  cout << is_copy_constructible_v<unique_ptr<MyClass>> << '\n';      // false 
  cout<< is_move_constructible_v<unique_ptr<MyClass>> <<"\n";        //true
  cout<< is_copy_assignable_v<unique_ptr<MyClass>> <<'\n';          //false
  cout<< is_move_assignable_v<unique_ptr<MyClass>> <<'\n';          //true

 

 
}
