#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

template<typename Container, typename F>
void foreach (Container c, F op)
{
    for(auto& data : c)
    {
        op(data);
    }                    
}

using FPointer=void (*)(int );
void print(int data)
{
  cout << data << " ";
}



struct IntPrinter {
    void operator() (int data) const {  
      cout <<data << ",";
    }
};


template<typename T>
struct Printer {
    void operator() (T data) const {  
      cout <<data << ",";
    }
};


template<typename T>
bool compare(T x, T y) { 
    return x > y; 
}

template<typename T>
class Comparer {
public: 
    bool operator() (T  x, T y) { 
        return x > y; 
    }
};

/*
struct ___LambdaXXXX {    
    void operator() (int data) const {  
      cout << data <<"-";
    }
};

template<typename T>
struct ___LambdaXXXX {    
    void operator() (T data) const {  
      cout << data <<"-";
    }
};
*/

int main()
{
    vector v = { 7,2,8,4,3 };

     FPointer p=print;
     p(100);
     foreach(v,p);   //函数指针 
     cout<<endl;

 
    IntPrinter intPter;
    intPter(100);

     Printer<int> pobj;
     pobj(100);

     foreach(v, pobj); //函数对象 inline
     foreach(v, Printer<int>{});
     cout<<endl;


     // Labmda表达式
    foreach(v,  [] (auto data)  { cout << data <<",";}  );
    cout<<endl;

    //foreach(v, ___LambdaXXXX{});

    // 函数指针
    sort(v.begin(),v.end(), compare<int>);                                   
    // 函数对象
    sort(v.begin(),v.end(), Comparer<int>{});
    sort(v.begin(),v.end(), std::greater<int>{});  


    // Labmda表达式
    sort(v.begin(),v.end(), [](auto x, auto y) -> bool { return x > y; });   

    foreach(v,  [] (auto data)  { cout << data <<"-";});

    // //foreach(v, ___LambdaXXXX{} );

    // cout<<endl;

    // foreach(v,[](auto data) {                  
    //                 cout << data <<"-";
    //             });
    // cout<<endl;
  
    
}
