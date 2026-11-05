#include <iostream>
using namespace std;

//void* T::operator new  ( std::size_t count );
//void* T::operator new[]( std::size_t count );

class MyClass{
public:
  MyClass(const string& s):_s(s){
    
  }
  MyClass(){ cout<<"ctor"<<endl;}
  ~MyClass(){ cout<<"dtor"<<endl;}
  void process(){
    cout<<_s<<endl;
  }

  void* operator new(size_t size)   {
    cout<<"class new "<<endl;

    data++;
    return ::operator new(size);
  } 
  void operator delete(void* p)  {
    cout<<"class delete "<<endl;
    ::operator delete(p); 

    data--;
  } 

  static int data;
private:
  string _s;
};

int MyClass::data=0;




int main()
{
  
  MyClass* myObject = new MyClass{"Software"};  
  myObject->process();             
  delete myObject;

  cout<<"-----------"<<endl;
  MyClass* p = ::new MyClass{}; 
  ::delete p;
 

}




