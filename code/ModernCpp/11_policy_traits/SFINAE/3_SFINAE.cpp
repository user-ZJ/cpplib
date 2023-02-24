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

template <typename T, typename = typename T::type> //SFINAE way for int
int process(const T& t) 
{
    int data=100;
    cout<<"const T& "<<endl;

    return data;
}



int main(){

    Widget w{100};
    process(w);

    int data;
    process(data);

}