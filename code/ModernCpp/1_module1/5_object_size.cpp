#include <iostream>
using namespace std;



class C1
{
    int i;    //4 byte  
    char c1;  //1 byte
    char c2;  //1 byte

public:
    virtual void process1()
    {
        i++;
    }

    virtual void process2()
    {
        i++;
    }
    
}; 




class C2
{
    char c1;  //4
    int i;    //4
    char c2;  //4
};//12 byte 




class C3
{
    int i1; //4
    char c1;//1

    int i2; //4
    char c2; //1

    int i3;
    char c3;
    
    int i4;
    char c4; 
   
};//32


class C4
{
    int i1; //4byte 
    int i2; 
    int i3;
    int i4;

    char c1;
    char c2; 
    char c3;
    char c4; 
};//20


//最多以pack size对齐
#pragma pack(8)
struct C5 {
    char a; // 1 byte
    int b; // 4 byte
    char c, d, e; // 3 byte
}; 
#pragma


int main()
{


  cout<<sizeof(C1)<<endl;// 输出8
  cout<<sizeof(C2)<<endl;// 输出12
  cout<<sizeof(C3)<<endl;// 输出32
  cout<<sizeof(C4)<<endl;// 输出20
  cout<<sizeof(C5)<<endl;// 10? 


}





