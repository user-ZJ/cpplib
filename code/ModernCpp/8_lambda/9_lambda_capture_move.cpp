
#include <iostream>
#include <vector>
#include <memory>

using namespace std;



struct Widget {

    Widget()
    {
        cout<<"ctor"<<endl;
    }

    Widget(const Widget& rhs)=delete;
	Widget& operator=(const Widget& rhs)=delete;	



    Widget(Widget&& rhs) noexcept
    { 

        cout<<"move ctor"<<endl; 
    }


    
    Widget& operator=(Widget&& rhs)	noexcept	
    {	
        cout<<"move assignment"<<endl;	
        return *this; 			
    }
    

    ~Widget(){
        cout<<"dtor "<<endl;

    }


    void process() const {
        cout<<"process"<<endl;
    }



};


int main()
{
	Widget w1;
    Widget w2;

//    {
//         auto lambda1 = [w1] () 
//         {
//             w1.process();
//         };

//         lambda1();
//    }
	
    cout<<"-----"<<endl;

     {
        auto lambda2 = [ w=std::move(w2)] () 
        {
            w.process();
        };

        lambda2();
   }

    cout<<"-----"<<endl;

    

     {
        unique_ptr<Widget> upw{new Widget()};

        auto lambda3 = [ upw2=std::move(upw) ] () 
        {
            upw2->process();
        };

        lambda3();
   }
	
    cout<<"-----"<<endl;

    {
        unique_ptr<Widget> u{new Widget()};

        Widget* p=u.release();
        auto lambda4= [ p ] () 
        {
            unique_ptr<Widget> up{p};
            up->process();
        };

        lambda4();
   }
    cout<<"-----"<<endl;
	
}
