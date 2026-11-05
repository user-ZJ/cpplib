#include <functional>
#include <iostream>
#include <vector>

using namespace std;
using namespace std::placeholders;

template<typename Container, typename Func>
void foreach (Container c, Func op)
{
    for(auto& data : c)
    {
        op(data); 
    }                    
}

void print1(int data, string prefix )
{
    cout<<prefix << data ;
    
}


void print2(int data, string prefix, string postfix )
{
     cout<<prefix << data <<postfix ;
}

bool compare(int data1, int data2, bool is_abs )
{
    if(is_abs)
    {   return abs(data1)> abs(data2);}
    else
    {   return data1> data2;}

}

int main()
{

    vector<int> v = { 8,-5,2,-4,7,-1,9 };

    //返回一个函数对象（类型为__bind)
    auto binder=std::bind(print1, _1, " * ");
    binder(100); // print1(100,"*");
    cout<<endl;

    cout<<sizeof(binder)<<endl;


    //绑定单参数的函数对象
    foreach(v, binder);
    foreach(v, std::bind(print1, _1, " * "));
    cout<<endl;



    //绑定两参数的函数对象
    foreach(v, std::bind(print2, _1, "[", "] "));
    cout<<endl;


    sort(v.begin(),v.end(), std::bind(compare,_1,_2, true));

    
    sort(v.begin(),v.end(), [](int data1, int data2){
        return compare(data1,data2,true);
    });
    

    sort(v.begin(),v.end(), std::bind(compare,_1,_2, false));
    foreach(v, std::bind(print2, _1, "{", "} "));
    cout<<endl;
    cout<<"\nlambda: "<<endl;


    foreach(v, [](auto data){ cout<<"{" << data <<"} ";});
    
    cout<<endl;
}
