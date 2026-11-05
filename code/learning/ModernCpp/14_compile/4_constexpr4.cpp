#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <forward_list>

using namespace std;

constexpr bool isPrime (unsigned int p)
{
    for (unsigned int d=2; d<=p/2; ++d) {
        if (p % d == 0) {
            return false; 
        }
    }
    return p > 1; 
}

template<int size, bool = isPrime(size)>
struct Widget
{

};

template<int SZ>
struct Widget<SZ, true>
{
    int data;

    void print(){
        cout<<data<<endl;
    }
};

template<int SZ>
struct Widget<SZ, false>
{
    double value;

    void process(){
        cout<<value<<endl;
    }
};

int main(){
    Widget<10> w1;
    Widget<17> w2;

    
    w1.value=100.234;
    w1.process();
    
    w2.data=300;
    w2.print();

}