#include <iostream>
using namespace std;


class Point{
public:
  Point(int x, int y, int z):x(x),y(y),z(z){ 
    cout<<"ctor"<<endl;
  }
  ~Point(){ cout<<"dtor"<<endl;}

  virtual void print(){
    cout<<x<<" "<<y<<" "<<z<<endl;
  }

  int x;
  int y;
  int z;
};


int main()
{
  
   cout<<sizeof(Point)<<endl;

   void* memory = std::malloc(sizeof(Point));

  int* p1=(int*)memory;
  for(int i=0;i<6;i++)
  {
    cout<<*p1<<" ";
    p1++;
  }
  cout<<endl;

   Point* myObject = ::new (memory) Point{100,200,300};
   myObject->print();

  long* p=(long*)memory;
  cout<<*p<<endl;
  p++;

  int* pi=reinterpret_cast<int*>(p);
  cout<<*pi++<<endl;
  cout<<*pi++<<endl;
  cout<<*pi<<endl;
  
   myObject->~Point();
   
   std::free(memory);

 

  
   
}




