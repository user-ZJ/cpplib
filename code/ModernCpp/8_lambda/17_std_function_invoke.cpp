#include <utility>
#include <functional>
#include <iostream>
#include <vector>

using namespace std;



class MyClass {
  public:

  int value;

    void process(int data){
        cout<<"MyClass process: "<<data<<endl;

        value+=data;
    }

    static void sprocess(int data){
      cout<<"MyClass static process: "<<data<<endl;
    }
};

void process(int data){
        cout<<"process function pointer: "<<data<<endl;
}

struct Processor {
    void operator() (int data) const {  
      cout  << "Processor Functor: "<< data <<endl;
    }
};







int main()
{
    MyClass mc{100};

    cout<<"invoke---------"<<endl;
    {
      std::invoke(process, 10 );
      std::invoke(Processor{}, 20 );
      cout<<mc.value<<endl;
      std::invoke(&MyClass::process, &mc, 30 );//mc.print(30);
      cout<<mc.value<<endl;
      std::invoke(&MyClass::process, mc, 30);
      cout<<mc.value<<endl;
      std::invoke(&MyClass::sprocess,40);
      std::invoke(
        [](int data){cout<<"lambda:"<<data<<endl;}, 
        50);
      
      
    }


    cout<<"function---------"<<endl;
    {
      std::function func1=process;
      std::function func2=Processor{};
      std::function<void(MyClass, int)> func3_1=&MyClass::process;
      std::function<void(MyClass&, int)> func3_2=&MyClass::process;
      std::function<void(MyClass*, int)> func3_3=&MyClass::process;
      std::function func4=&MyClass::sprocess;
      std::function func5=[](int data){cout<<"lambda:"<<data<<endl;};

      //mc.process(10); // process(&mc, 10);

      func1(100);
      func2(200);
      cout<<mc.value<<endl;

      func3_1(mc,300);
      cout<<mc.value<<endl;

      func3_2(mc,300);
      cout<<mc.value<<endl;

      func3_3(&mc,300);
      cout<<mc.value<<endl;
      
      func4(400);
      func5(500);

      std::invoke(func5,500);
      
    }
    
}

