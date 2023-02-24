#include <iostream>
using namespace std;


class Widget {


};

Widget getWidget(){
    Widget w;
    return w;
}

template <typename T>
void invoke(T&& obj)
{
    cout<<is_lvalue_reference<T>::value<<endl; //Widget&
    cout<<is_lvalue_reference<T&&>::value<<endl; //Widget& && -> Widget&
    cout<<is_lvalue_reference<decltype(obj)>::value<<endl; //Widget&
    cout<<is_lvalue_reference<T&>::value<<endl; //永远左值 Widget& & 

    cout<<endl;

    cout<<is_rvalue_reference<T>::value<<endl;
    cout<<is_rvalue_reference<T&&>::value<<endl;
    cout<<is_rvalue_reference<decltype(obj)>::value<<endl;
    cout<<is_rvalue_reference<T&>::value<<endl; //永远左值

}

// & &   --> &
// & &&  --> &
// && &  --> &
// && && --> &&

int main(){

    cout<<std::boolalpha;
    Widget w;
    invoke(w); // w: Widget&   ->  invoke(Widget& && obj) --> invoke(Widget& obj)
    cout<<"-------"<<endl;

    invoke(getWidget());
    
    cout<<"-------"<<endl;

    {
        auto w1=w;
        auto& w2=w;
        auto&& w3=w;
        //auto& & w4=w; 没有引用的引用

        cout<<is_lvalue_reference<decltype(w)>::value<<endl;
        cout<<is_lvalue_reference<decltype(w1)>::value<<endl;
        cout<<is_lvalue_reference<decltype(w2)>::value<<endl;
        cout<<is_lvalue_reference<decltype(w3)>::value<<endl;

        cout<<endl;

        cout<<is_rvalue_reference<decltype(w)>::value<<endl;
        cout<<is_rvalue_reference<decltype(w1)>::value<<endl;
        cout<<is_rvalue_reference<decltype(w2)>::value<<endl;
        cout<<is_rvalue_reference<decltype(w3)>::value<<endl;
    }

    cout<<"-------"<<endl;

    {
        auto w1=getWidget();
        const auto& w2=getWidget();
        auto&&  w3=getWidget();

        cout<<is_lvalue_reference<decltype(w)>::value<<endl;
        cout<<is_lvalue_reference<decltype(w1)>::value<<endl;
        cout<<is_lvalue_reference<decltype(w2)>::value<<endl;
        cout<<is_lvalue_reference<decltype(w3)>::value<<endl;

        cout<<endl;

        cout<<is_rvalue_reference<decltype(w)>::value<<endl;
        cout<<is_rvalue_reference<decltype(w1)>::value<<endl;
        cout<<is_rvalue_reference<decltype(w2)>::value<<endl;
        cout<<is_rvalue_reference<decltype(w3)>::value<<endl;
    }
}

