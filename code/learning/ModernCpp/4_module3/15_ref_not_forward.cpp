#include <iostream>
using namespace std;




template<typename T>
class Vector{

public:


    void process(T&& obj)
    {
        cout<<is_rvalue_reference<decltype(obj)>::value<<endl;

    }

    template<typename U>
    void process(U&& obj)
    {
        cout<<is_rvalue_reference<decltype(obj)>::value<<endl;

    }
};


template<typename T>
void process(T&& obj)
{
    cout<<is_rvalue_reference<decltype(obj)>::value<<endl;

}

int main()
{
   cout<<std::boolalpha;
   string s="forward";

    Vector<string> v;
    v.process("hello"s);
    v.process(s);

    process("hello"s);
    process(s);


}         

