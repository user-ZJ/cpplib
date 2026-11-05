#include <iostream>
#include <vector>
using namespace std;

int alloc_times = 0; 
int dealloc_times = 0; 
int allocated = 0; 


class MyClass{
 
private:
  double x,y,z;
};



void* operator new(size_t size)  { 
  void* p = std::malloc(size); 
  std::cout << "allocated " << size << " byte(s)\n"; 

  allocated+=size;
  alloc_times++;
  return p; 
} 
 
void operator delete(void* p) noexcept  { 
  std::cout << "deleted memory\n"; 

  dealloc_times++;
  return std::free(p); 
}

void* operator new[](size_t size)   {
  void* p = std::malloc(size); 
  allocated+=size;
  alloc_times++;

  std::cout << "allocated " << size << " byte(s) with new[]\n"; 
  return p; 
} 
void operator delete[](void* p) noexcept  { 
  std::cout << "deleted memory with delete[]\n"; 
   dealloc_times++;
  return std::free(p); 
}

int main()
{
  
  {
    MyClass* v=new MyClass[10];//24*10=240

    cout<<allocated<<endl;

    for(int i=0;i<10;i++)
    {
      MyClass* pc=new MyClass();//10
      delete pc;
    }
    
    delete[] v;

  }



  cout<<allocated<<endl;
  cout<<alloc_times<<endl;
  cout<<dealloc_times<<endl;



}




