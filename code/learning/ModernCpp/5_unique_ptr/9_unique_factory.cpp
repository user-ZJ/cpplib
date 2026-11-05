#include <memory>
#include <iostream>

using namespace std;


class Widget{
public:	
    Widget() { cout<<"ctor"<<endl;}
    ~Widget(){ cout<<"dtor"<<endl;}

    Widget(const Widget& rhs){ cout<<"copy ctor"<<endl;}	
    Widget(Widget&& rhs){ cout<<"move ctor"<<endl; }	

    Widget& operator=(Widget&& rhs)	{	
        cout<<"move assignment"<<endl;	
        return *this; 			
    }
	Widget& operator=(const Widget& rhs){
        cout<<"copy assignment"<<endl;
        return *this;
    }	

};

// Widget& createWidget(){

//     Widget* w=new Widget();
//     //....

//     return *w;
// }


// Widget createWidget(){

//     Widget w;
//     //....

//     return w;
// }

// Widget* createWidget(){

//     Widget* pw=new Widget();
//     //....

//     return pw;
// }

unique_ptr<Widget> createWidget(){

    unique_ptr<Widget> upw=make_unique<Widget>();
    //....

    return upw;
}

int main()
{
    unique_ptr<Widget> upw2;

    upw2=createWidget();

}