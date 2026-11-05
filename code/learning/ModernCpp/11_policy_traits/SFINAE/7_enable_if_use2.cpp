#include <iostream>
#include <vector>

using namespace std;

template<typename T>
class Widget
{
public:
    void process(T const& x){
        cout<<"T const& x"<<endl;
    }

    
    template<typename U=T, typename V= enable_if_t<!is_reference_v<U>>>
    void process(T&& x)
    {
        cout<<"T && x"<<endl;
    }

    // template<typename U=T>
    // typename enable_if<!std::is_reference<U>::value>::type process(T&& x)
    // {
    //     cout<<"T && x"<<endl;
    // }
};

int main(){

    {
        Widget<int> w;

        int data=100;
        w.process(data);
        w.process(100);
    }

    {
        Widget<int&> w;

        int data=100;
        w.process(data);
    }



}