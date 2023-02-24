#include <iostream>
#include <memory>
 
using namespace std;
#include <memory>
#include <iostream>

/*
template<typename Y, typename Deleter>
shared_ptr(unique_ptr<Y,Deleter>&& u);
*/

unique_ptr<string> process()
{
    return make_unique<string>("hello");
}

int main()
{
   
    shared_ptr<string> sp1 = process(); 


    auto up = make_unique<string>("Hello World");

    shared_ptr<string> sp2 = move(up); 
    //shared_ptr<string> sp2 = up; 

     if(sp2.unique())
         cout<<"only 1 count"<<endl;
         

}

