#include <iostream>
using namespace std;

struct Widget{

    using type=int;

    int data;

    void increase(){
        data++;
    }
};



int process(...)
{
    int data=100;
    cout<<"..."<<endl;

    return data;
}

template <typename T> 
int process(const T& t, typename T::type* p=nullptr) //SFINAE way for int
{
    int data=100;
    cout<<"const T& "<<endl;

    return data;
}



int main(){

    Widget w{100};
    process(w); // int process(const T& t, int* p=nullptr)

    int data;
    process(data); // int process(const T& t, int::type* p=nullptr)

}