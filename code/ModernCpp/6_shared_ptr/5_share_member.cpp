#include <iostream>
#include <memory>
 
using namespace std;

struct Widget{  
    
    int* data; 

    ~Widget(){
        cout<<"~Widget()"<<endl;
    }
};


void process(Widget*)
{
}




int main()
{   
    {
        Widget* w=new Widget();

        shared_ptr<Widget> spw1{w};


        shared_ptr<Widget> spw2{spw1};

        process(spw1.get());
    }
    {
  

        shared_ptr<Widget> spw(new Widget());
        
        cout<<spw.use_count()<<endl;

        
        //shared_ptr<int> spd{spw->data};
         shared_ptr<int> spd{spw, spw->data};
        
        

         cout<<spd.use_count()<<endl;
         cout<<spw.use_count()<<endl;
         spw.reset();
         cout<<"after reset..."<<endl;
         cout<<spd.use_count()<<endl;
         cout<<spw.use_count()<<endl;

     
    }

 
}


//创建ptr的指针，与r共享所有权
// template< class Y >
// shared_ptr( const shared_ptr<Y>& r, element_type* ptr ) noexcept;
