#include <iostream>
#include <string>
using namespace std;





struct Point{
    double x;
    double y;
};




class Rectangle1
{
    Point leftUp;//值语义
    int width;
    int height;

};

/*
class Rectangle1
{
    double x;
    double y;
    int width;
    int height;

};*/


class Rectangle2
{
    Point* m_leftUp; //堆指针

    int m_width;
    int m_height;
public:
    Rectangle2(double x, double y, int width,int height):
        m_leftUp(new Point{x,y}),
        m_width(width),
        m_height(height)
        {

            
        }

    ~Rectangle2()
    {
        delete m_leftUp;
    }
};

class Rectangle3
{
    Point& leftUp;//引用
    int width;
    int height;

};




class MyShape
{
    Rectangle1 r1;//24 byte
    Rectangle2 r2;//16 byte
    Point p;//16 byte
    int x;//4 byte
    int y;//4 byte
    int * data;//8 byte

};

int main(){

    Rectangle1 r1;

    Rectangle2 r2(10,20,100,200);

    cout<<sizeof(Point)<<endl;
    cout<<sizeof(Rectangle1)<<endl;
    cout<<sizeof(Rectangle2)<<endl;
    cout<<sizeof(Rectangle3)<<endl;
    cout<<sizeof(MyShape)<<endl;

    
}