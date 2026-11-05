
#include <memory>
#include <iostream>
using namespace std;

class Widget{
public:
    int m_x;
    int m_y;
    int m_z;


    Widget(int x,int y , int z):
        m_x(x), m_y(y),m_z(z)
    {}

    void print(){
        cout<<m_x<<","<<m_y<<","<<m_z<<endl;
    }

    ~Widget()
    {
        cout<<"Widget dtor"<<endl;
    }
};


unique_ptr<Widget> getWidgetPtr()
{
    unique_ptr<Widget> w(new Widget(100,200,300));


    return w;
}

/*
Widget* getWidgetPtr()
{
    Widget* w(new Widget(100,200,300));

    ///....
    return w;
}*/



void process(Widget* p)
{
   
    p->print();
}



int main()
{

 
          
    {
        //Widget w(1,2,3);
        Widget* p=new Widget(1,2,3);
        //unique_ptr<Widget> w1{&w}; //delete &w

        p->print();
       

        unique_ptr<Widget> w2{p};

         w2->print();
    
    }

    
    {
        
        Widget* pw=new Widget(1,2,3);
        unique_ptr<Widget> w1{pw};
        w1->print();

        unique_ptr<Widget> w2 { new Widget(*w1)} ;

        auto w3= unique_ptr<Widget>(new Widget(10,20,30));
        auto w4=getWidgetPtr();
        unique_ptr<Widget> w5=std::move(w1); //移动构造
        if(w1==nullptr) cout<<"w1 is nullptr"<<endl;
        // w2->print();

        w3.swap(w4);
        w3->print();
        w4->print();

 

        w2.reset(new Widget(11,22,33)); 
        
        w3.reset(); // 等价 w3=nullptr; 

        if(w2!=nullptr)
        {
            cout<<"w2 is not null"<<endl;
        }
        else 
        {
            cout<<"w2 is null_ptr"<<endl;
        }

    }
    cout<<"----------"<<endl;


    {
        auto w1 = make_unique<Widget>(1000,2000,3000);
        auto w2 = make_unique<Widget>(100,200,300);


        Widget* rawp1=w1.get(); 
        //delete rawp1;

        process(rawp1); //process(w1.get());
        w1->print();


         Widget* rawp2=w2.release();//释放所有权
         if(w2==nullptr) cout<<"w2 is null_ptr"<<endl;
         delete rawp2; //必须负责释放d

    }
    
    


    
}
