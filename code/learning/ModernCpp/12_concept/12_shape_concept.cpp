#include <iostream>

using namespace std;

template <typename T>
concept Shape = requires(T t, int i, double d) 
{
    { t.getArea() } -> std::integral;
    t.draw(i, d) ;
};


struct Line
{
    void draw(int data, double d) const{
        cout<<"Line draw :"<<data<<endl;
    }

    int getArea() const
    {
        cout<<"Line getArea"<<endl;
        return 300;
    }
	

};


struct Rect
{
    void draw() const{
        cout<<"Rect draw"<<endl;
    }

    int getArea() const
    {
        cout<<"Rect getArea"<<endl;
        return 300;
    }
	

};


struct Square
{
    void draw() const {
        cout<<"Square draw"<<endl;
    }

    int getArea() const
    {
        cout<<"Square getArea"<<endl;
        return 300;
    }
	

};

template<Shape T>
void process(T& shape)
{
    
	shape.getArea();
    shape.draw(30,3.14);
}


static_assert( Shape<Line>);

int main(){
    
    Line shape;
    process( shape);

 
    //string s="hello"s;
    //process(s);
}





