#include <functional>  
#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace std::placeholders;

class Widget {
  public:
    int  no;
    string name;
    Widget (int _no, const std::string& _name ) 
        : no(_no), name(_name) {
    }

    void print(){
        cout<<no<<": "<<name<<" "<<endl;
    }
};


void update(Widget& w)
{
    w.no+=1;
    w.name="["+w.name+"]";
}



int main()
{
    std::vector<Widget> widgets;  

    Widget w1(100,"C++");
    Widget w2(200,"Java");

    update(w1);
     w1.print();
  
    auto binder1=bind(&update, w1);
    binder1();
    w1.print();


    auto binder2=bind(&update, std::ref(w2));
    binder2();
    w2.print();

}
