#include <iostream>
using namespace std;

class Point{
public:
    int x=10;
    int y=20;//实例变量
    inline static int data=100;

    void process1()
    {
        cout<<this->x<<endl;
        std::cout<<x<<endl;
        y++;

        process2(100);

        process3();

    }

    void process2(int d)
    {

    }

    static void process3()
    {
        //x++;
        data++;

       
    }

};

/*
int Point::data=10;

struct Point {
    int x;
    int y;
};

void process1(Point * this)
{
    cout<<this->x<<endl;
    std::cout<<this->x<<endl;
    this->y++;


    process2(this, 100);

    process3();
   
}

void process2(Point * this, int d)
{
    
}

void process3()
{
    //x++;
    Point::data++;
    
}
*/



int main(){

    Point pt;
    pt.process1(); // process(&pt); JMP 0x000064
    pt.process3(); // Point::process3()
    Point::process3();

    Point* p=new Point();
    p->process1(); //process(p);
    p->process3(); // Point::process3();
    delete p;
}