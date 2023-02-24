#include <iostream>
using namespace std;

struct Base{

    Base(){
        process(); //静态绑定
        cout<<"Base()"<<endl;
    }

     virtual ~Base(){
        cout<<"~Base()"<<endl;
        //process();//静态绑定
    }

    // 0x0023723880
    void invoke(){
        process(); //动态绑定
        cout<<"Base.invoke()"<<endl;
    }


    //0x008234960
    virtual void process(){
        cout<<"Base.process()"<<endl;
    }

};

struct Sub:  Base{

    int data;

    Sub():Base(){
        data=100;
    }

    //0x008234100
    void process() override {

        //Base::process(); ////静态绑定 JMP 0x008234960
        cout<<"Sub.process()"<<endl;
        data++;
    }

    //0x002342160
    void invoke(){

        cout<<"Sub.invoke()"<<endl;
    }

    ~Sub(){
        cout<<"~Sub()"<<endl;
    }
};

int main()
{
    
    Base * pb1=new Base();
    pb1->process();//动态绑定 JMP *(pb1->vptr+2*8) 二次指针间接运算
    pb1->invoke();//静态绑定 JMP 0x0023723880
    

    Sub * ps=new Sub();
    ps->process();//动态绑定 JMP vfunc(p2) 二次指针间接运算
    ps->invoke();//静态绑定 JMP 0x002342160

    Base* pb2=ps;
    pb2->process(); //动态绑定 JMP vfunc(p2) 二次指针间接运算
    pb2->invoke(); //静态绑定 JMP 0x0023723880

    pb2->Base::process();//虚函数静态绑定 JMP 0x008234960

    cout<<"----------"<<endl;
    delete pb1;
    delete pb2;

}


