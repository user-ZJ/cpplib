#include <iostream>
#include <algorithm>
using namespace std;

class Point{
public:
  Point(int x, int y, int z):x(x),y(y),z(z){ cout<<"ctor"<<endl;}
  
  Point(const Point& ){
    cout<<"copy ctor"<<endl;
  }

   Point( Point&& ) noexcept{
    cout<<"move ctor"<<endl;
  }
  ~Point(){ cout<<"dtor"<<endl;}

  void print(){
    cout<<x<<" "<<y<<" "<<z<<endl;
  }

  int x;
  int y;
  int z;
};


int main()
{

  {
    void* memory = std::malloc(3*sizeof(Point));

    Point* myObject = reinterpret_cast<Point*>(memory);

    std::uninitialized_fill_n(myObject, 3, Point{100,200,300});
    cout<<"----"<<endl;
    
  
    std::destroy_n(myObject,3); //std::destroy_at(myObject);

    std::free(memory);
  }

 
}




