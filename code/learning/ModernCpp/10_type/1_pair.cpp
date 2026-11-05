#include <utility>
#include <string>
#include <complex>
#include <tuple>
#include <iostream>

using namespace std;



class Widget{

    
public:	

    int m_data;

    Widget(int data):m_data(data) { cout<<"ctor"<<endl;}
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

Widget getWidget(){
    Widget w(100);
    return w;
}
 
int main()
{

 
    std::pair<int, double> p1{42,3.1415};
    cout<<p1.first<<","<<p1.second<<endl;

    
    auto p2=make_pair(42, 3.1415); //std::pair<int, double> p2{42, 3.1415};

    std::pair p3{42, 3.1415};
    
    std::pair<char, int> p4{p3};
    cout<<p4.first<<","<<p4.second<<endl;
 
    pair<string, Widget> p5{"hello"s, getWidget()};


    pair p6=p5;
    pair p7=std::move(p6);

}