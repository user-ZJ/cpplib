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

// template <typename T> 
// int  process(const T& t) //NO SFINAE
// {
//     typename T::type data=100;
//     cout<<"const T& "<<endl;

//     return data;
// }

template <typename T> 
typename T::type process(const T& t) //SFINAE away for int     int::type
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