#include <iostream>

using namespace std;

class MyClass {
    int data;


public:
     MyClass(int d):data(d){

     }

    // == , !=
    bool operator== (const MyClass& rhs) const
    {
        return data == rhs.data; 
    }

    // <, <=, >, >=
    auto operator<=> (const MyClass& rhs) const 
    {
        return data <=> rhs.data; 
    }
};

int main(){

    int d1=300;
    int d2=200;
    auto result= d1<=>d2;

    if(result==0) cout<<"d1等于d2"<<endl;
    if(result<0) cout<<"d1小于d2"<<endl;
    if(result>0) cout<<"d1大于d2"<<endl;
    



    MyClass c1(100);
    MyClass c2(200);

    cout<<std::boolalpha;

    cout<< (c1==c2) <<endl;
    cout<< (c1!=c2) <<endl;
    cout<< (c1>c2) <<endl;
    cout<< (c1>=c2) <<endl;
    cout<< (c1<c2) <<endl;
    cout<< (c1<=c2) <<endl;

   
  
    
}