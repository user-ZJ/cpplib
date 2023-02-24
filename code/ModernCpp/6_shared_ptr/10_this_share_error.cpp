#include <memory>
#include <iostream>

using namespace std;




class Widget {

public:
    
    shared_ptr<Widget> getWidget() {

        return shared_ptr<Widget>(this);  // 

    }
};


// shared_ptr<Widget> getWidget(Widget * this) {

//         return shared_ptr<Widget>(this);  // 
// }


int main()
{
     {
        Widget* p=new Widget();

        shared_ptr<Widget> pw{p};
        
        shared_ptr<Widget> s=pw->getWidget(); //  
        // shared_ptr<Widget> s= getWidget( pw.get());
        // shared_ptr<Widget> s= shared_ptr{p};

    }

    {
        Widget w;
        shared_ptr<Widget> s=w.getWidget();
    }

    {
        Widget* pw=new Widget();
        shared_ptr<Widget> s=pw->getWidget();
        delete pw;
    }

   

}
