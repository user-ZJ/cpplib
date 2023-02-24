
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

void process(shared_ptr<Widget> sp1, shared_ptr<Widget> sp2)
{

}

void invoke(){
    
}

int main()
{
    {
        // 不好：可能会泄漏
        // process(shared_ptr<Widget>(new Widget()), invoke());

        // p1=new Widget();
        // p2=new Widget();
        // shared_ptr<Widget>(p1);
        // shared_ptr<Widget>(p2);

        // p1=new Widget();
        // shared_ptr<Widget>(p1);
        // p2=new Widget();
        // shared_ptr<Widget>(p2);
        // process(p1, p2);
    }


    {
        // 好多了，但不太干净
        shared_ptr<Widget> sp1(new Widget()); 
        shared_ptr<Widget> sp2(new Widget());
        process(sp1, sp2);
    }

    {
        // 最好，也很干净
        process(make_shared<Widget>(), make_shared<Widget>());
    }

}