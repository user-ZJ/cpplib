#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
using namespace std;


template <class InputIterator, class Function>
inline void foreach(InputIterator first,InputIterator last,Function f) {
  for (; first != last; ++first)
    f(*first);
}


using FPointer=void (*)(int, int );

template<typename T>
bool Compare(T x, T y) { 
    return x > y; 
}

template<typename T>
struct Greater {
    bool operator() (T  x, T y) { 
        return x > y; 
    }
};

struct Less {
    bool operator() (int  x, int y) { 
        return x > y; 
    }
};


// template <class _RandomAccessIterator, class _Compare>
// inline void sort(   _RandomAccessIterator __first, 
//                     _RandomAccessIterator __last, 
//                     _Compare __comp)

int main()
{
    vector<int> c = { 1,2,3,4,5 };
  
    foreach(c.begin(), c.end(),[] (auto data) { cout << data <<", "; });

    // 函数指针
    sort(c.begin(),c.end(), Compare<int> );  

    // 函数对象
    sort(c.begin(),c.end(), Greater<int>{});  

    // Labmda表达式
    sort(c.begin(),c.end(),   [](int x, int y) { return x > y; });   

    foreach(c.begin(), c.end(),[] (auto data) { cout << data <<", "; });
    cout<<endl;



    
}
