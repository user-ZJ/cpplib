#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

using namespace std;


class Widget {
  public:
    int  no;
    string name;
    Widget (int _no, const std::string& _name ) 
        : no(_no), name(_name) {
    }

    void print(){
        cout<<"["<<no<<"] "<<name<<" ";
    }
};

void value_default()
{
    vector<Widget> widgets;  

    Widget w1(100,"C++");
    Widget w2(200,"Java");
    
    widgets.push_back(w1);    
    widgets.push_back(w2);
    widgets.push_back(w1);
    for(auto& widget: widgets)
    {
        widget.print();
    }

    cout<<endl;
    
    w1.no++;
    w1.name="GO";

    for(auto& widget: widgets)
    {
        widget.print();
    }

    cout<<endl;

}

void ref_wrap()
{
    std::vector<std::reference_wrapper<Widget>> widgets;  

    Widget w1(100,"C++");
    Widget w2(200,"Java");
    
    widgets.push_back(w1);    
    widgets.push_back(w2);
    widgets.push_back(w1);
    for(auto& widget: widgets)
    {
        widget.get().print();
    }

    cout<<endl;
    
    w1.no++;
    w1.name="GO";

    for(auto& widget: widgets)
    {
        widget.get().print();
    }

    cout<<endl;

}

void share_wrap()
{
    std::vector<std::shared_ptr<Widget>> widgets;  

    auto w1=make_shared<Widget>(100,"C++");
    auto w2=make_shared<Widget>(200,"Java");
    
    widgets.push_back(w1);    
    widgets.push_back(w2);
    widgets.push_back(w1);
    for(auto& widget: widgets)
    {
        widget->print();
    }

    cout<<endl;
    
    w1->no++;
    w1->name="GO";

    for(auto& widget: widgets)
    {
        widget->print();
    }

    cout<<endl;

}

int main(){
    
    value_default();
    cout<<endl;
    ref_wrap();
    cout<<endl;
    share_wrap();
}