#include <memory>
#include <iostream>

using namespace std;


struct Widget{


    int m_data;

    Widget(int data=1):m_data(data) { cout<<"ctor"<<endl;}
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




int main()
{
    Widget* pw=new Widget[10];

    delete [] pw;

    //unique_ptr<Widget> upArr=unique_ptr<Widget>(new Widget[10]); //错误!


    //vector<Widget> vw{10};
   
    unique_ptr<Widget[]> upArr1 = std::unique_ptr<Widget[]>(new Widget[5]); 

    for(int i=0;i<5;i++)
    {
        cout<<upArr1[i].m_data<<" ";
    }

    cout<<"\n";


    unique_ptr<Widget[]> upArr2 = make_unique<Widget[]>(5);
    for(int i=0;i<10;i++) //内存越界，行为未定义
    {
        upArr2[i].m_data++;
    }

    for(int i=0;i<10;i++)
    {
        cout<<upArr2[i].m_data<<" ";
    }
    cout<<"\n";

}