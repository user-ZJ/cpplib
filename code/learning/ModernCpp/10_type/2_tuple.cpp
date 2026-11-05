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
 

tuple<int, string, double> get_book(int id)
{
    if (id == 100) return std::make_tuple(100, "C++ Programming Language", 100.5);
    if (id == 200) return std::make_tuple(200, "Effective Modern C++", 80.00);
    if (id == 300) return std::make_tuple(300, "Design Patterns", 45.5);
   
    return std::make_tuple(0,"",0.0);
}
 
int main()
{
    tuple book1 = get_book(100);
    cout  << std::get<0>(book1) << ", " << std::get<1>(book1) << std::get<2>(book1) << '\n';
 
    int id1;
    string name1;
    double price1;

    std::tie(id1, name1, price1) = get_book(200);
    cout<<id1<<", "<<name1<<", "<<price1<< '\n';
 

    auto [ id2, name2, price2 ] = get_book(300);
     cout<<id2<<", "<<name2<<", "<<price2<< '\n';

    tuple<int, Widget> t{ 100, getWidget()};

    tuple<int, Widget>* pt=new tuple<int, Widget> { 100, getWidget()};
 
}