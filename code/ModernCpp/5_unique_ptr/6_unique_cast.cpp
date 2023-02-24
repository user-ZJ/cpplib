#include <iostream>
#include <memory>
using namespace std;

class Base 
{ 
public:
    int a; 
    virtual void process()  { 
        std::cout << "Base.process()"<<endl;
    }
    
    virtual ~Base(){
        cout<<"~Base()"<<endl;
    }
};
 
class Sub : public Base
{
public:
    void process()  override{ 
        std::cout << "Sub.process()"<<endl; 
    }

    ~Sub() {
        cout<<"~Sub()"<<endl;

    }
};


 
int main(){

    {
        unique_ptr<Base> b1 = std::make_unique<Base>();
        b1->process();

        unique_ptr<Sub> s1 = std::make_unique<Sub>();
        s1->process();
    
        // Sub* ps=new Sub();
        // Base* pb=ps;
        // pb->process();
        unique_ptr<Base> b2 {std::move(s1)};  // Base* pb= ps;
        b2->process(); //智能指针维持多态性

  
    }
    cout<<"-----"<<endl;
    {
        unique_ptr<Base> upb =make_unique<Sub>();

        //unique_ptr<Sub> ups=(unique_ptr<Sub>)std::move(upb);
        
        // Base* pb=upb.release();
        // Sub* ps=dynamic_cast<Sub*>(pb);
        // unique_ptr<Sub>  ups{ps};

        unique_ptr<Sub>  ups( dynamic_cast<Sub*>(upb.release())); //Base* --> Sub*
        ups->process();
 
    }
    
}