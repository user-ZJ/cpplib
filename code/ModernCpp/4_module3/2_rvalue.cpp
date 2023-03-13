#include <iostream>
#include <numeric>
using namespace std;



void process(int&& data);
void process(int& data);

void process(int&& data)
{
    // 进函数后data是个左值
    cout<<"right value ref"<<endl;
    process(data); 
}


void process(int& data)
{
    cout<<"left value ref"<<endl;
}


int main()
{
    int x=100;
    int y=200;
    
    process(x);
    process(100);

    cout<<"-----"<<endl;

     process(++x);
     process(x++);
     //int*p1=&(++x);
     cout<<"-----"<<endl;

     process(x=300);
     process(x+y);
     cout<<"-----"<<endl;
  
    
     auto *p1=&("hello");
    cout<<p1<<endl;
    
    // auto *p2=&("hello"s);
    // cout<<p2<<endl;
    
    
}