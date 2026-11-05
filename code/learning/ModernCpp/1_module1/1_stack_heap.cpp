#include <iostream>
#include <vector>
#include <string>

using namespace std;


int data;


class Point{
public:
    int x{0};
    int y{0}; 


    void print()
    {
        cout<<x<<","<<y<<endl;
    }

    inline static int data=100;
};

//int Point::data=100;


void process()
{
    const int len=100;
    //常量栈数组
    int myarray[len];//alloca

}


void func()
{
    double data=3.14;

    Point pt{10,20};
    int x=10;
    int y=20;

    Point* p1=new Point{x,y};

    ///Point* p2=new Point{x,y};

    delete p1;
    ///delete p2;
}

int main()
{
    func();

    int x {10};
    int arr[]={1,2,3,4,5};//栈数组
    auto y=20;
    auto s1="hello"s;//自动类型推导 string
    int & data=x; //栈引用


    int* px =new int{10}; //堆指针对象
    int* parray=new int[5]{1,2,3,4,5}; //堆数组

    delete px;
    delete[] parray;

        
    {
        vector v{10,20,30,40,50};

        for(auto& s : v)
        {
            cout<<s<<endl;

        }
    }

    Point pt1{10,20};//栈对象
    Point pt2{100,200};

    
    Point* p3=new Point{10,20};//堆对象
    Point* p4=new Point{30,40};

    pt1=pt2;//stack->stack
    pt1.print();
    *p3=*p4;//heap->heap
    p3->print();
    pt2=*p4;//heap->stack
    pt2.print();
    *p4=pt1;//stack->heap
    p4->print();




    Point& pr1=*p3;//堆引用
    Point& pr2=pt1;//栈引用

    Point* p5=&pt1; //栈指针
    Point* p6=&pr1; //堆指针


    cout<<Point::data<<endl;
    cout<<p3->data<<endl;


    delete p3;
    delete p4;
    
}








