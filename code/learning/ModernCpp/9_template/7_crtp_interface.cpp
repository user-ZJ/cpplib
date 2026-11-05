#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <vector>
using namespace std;

// 基类参数T包含了子类编译时信息
template <typename T> 
class Base {
public:

  // Base(){
  //   sub=....;
  // }

    void process() { 
      sub()->process_imp(); //编译时分发
    }
    //如果不实现，类似纯虚函数
    void process_imp() { 
      cout<<"Base::process()"<<endl;
    }

    //将基类指针转型为子类T的指针
    T* sub() { return static_cast<T*>(this); } //  Base<T>*  ---> T*        Base<Sub1>* ---> Sub1* 

    ~Base()
    {
      //delete sub(); // static_cast<T*>(this);
      cout<<"~Base()"<<endl;

    }

    void destroy() 
    { 
        delete sub();// static_cast<T*>(this); 
    }

 
    // T* sub;
  
};


class Sub1 : public Base<Sub1> {


public:
    ~Sub1()
    {
      cout<<"~Sub1"<<endl;
    }

    void process_imp() { 
      cout<<"Sub1::process()"<<endl;
    }
};






template <typename T>
void invoke(Base<T>* pb)
{
  pb->process();
}


int main()
{


  Base<Sub1> *ps1=new Sub1();
  ps1->process();// process(ps1)


  invoke(ps1);

  //delete ps1;
  //delete ps1->sub();
  ps1->destroy();

}