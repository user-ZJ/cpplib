
#include <iostream>
#include <memory>
#include <vector>

using namespace std;



struct WidgetA{
    double x;
};

struct WidgetB{
    double y;
    double z;
};

struct WidgetC{
    double u;
    double v;
    double w;
};

template<class...Base>
struct Object: Base...{

};

// template<class T1>
// struct Object: T1{

// };

// template<class T1, class T2>
// struct Object: T1, T2{

// };

// template<class T1, class T2, class T3>
// struct Object: T1, T2, T3{

// };

template<class... T>
Object(T...)-> Object<T...>;



int main(){

    WidgetA a{1.1};
    WidgetB b{2.2,3.3};
    WidgetC c{4.4,5.5,6.6};

    Object obj{a,b,c}; // 
    //Object<WidgetA, WidgetB, WidgetC> obj{a,b,c};
    
    cout<<sizeof(obj)<<endl;
    cout<<obj.x<<endl;
    cout<<obj.y<<endl;
    cout<<obj.z<<endl;
    cout<<obj.u<<endl;
    cout<<obj.v<<endl;
    cout<<obj.w<<endl;

}