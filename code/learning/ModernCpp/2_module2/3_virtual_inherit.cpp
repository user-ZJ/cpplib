#include <iostream>
using namespace std;

struct Base1{
    double d1;
    double d2;
    double d3;
    double d4;
};
struct Sub1: virtual   Base1{
    double d5;
};

struct Sub2: virtual  Base1{
    double d6;

};

struct Sub3:   Sub1,   Sub2{
    double d7;
};



int main()
{
    cout<<sizeof(Base1)<<endl;//32
    cout<<sizeof(Sub1)<<endl;// 32+8=40       32(B1)+8(S1)+8(vb*)=48 
    cout<<sizeof(Sub2)<<endl;// 32+8=40       32(B1)+8(S2)+8(vb*)=48  
    cout<<sizeof(Sub3)<<endl; // 40+40+8=88   32(B1)+8(S1)+8(S2)+8(S1vb*)+8(S2vb*)+8(S3)=72

    Sub3 s3;
    
    s3.d1=3.14;//s3.Sub1::d1=3.14

}