
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

void process(unique_ptr<Widget> sp1, unique_ptr<Widget> sp2)
{

}



int main()
{
    {
        // 不好：可能会泄漏
        process(unique_ptr<Widget>(new Widget()), 
            unique_ptr<Widget>(new Widget()));

        // p1=new Widget();
        // p2=new Widget();
        // unique_ptr<Widget>(p1);
        // unique_ptr<Widget>(p2);

        // p1=new Widget();
        // unique_ptr<Widget>(p1);
        // p2=new Widget();
        // unique_ptr<Widget>(p2);
        // process(std::move(p1), std::move(p2));
    }


    {
        // 好多了，但不太干净
        unique_ptr<Widget> sp1(new Widget()); 
        unique_ptr<Widget> sp2(new Widget());
        process(std::move(sp1), std::move(sp2));
    }

    {
        // 最好，也很干净
        process(make_unique<Widget>(), make_unique<Widget>());
    }

}