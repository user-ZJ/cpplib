#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <forward_list>

using namespace std;




template<typename Iter>
void process (Iter start, Iter end )
{
 
    if constexpr( random_access_iterator<Iter>)
    {
       cout<<"处理 random_access_iterator"<<endl;
    }
    else if constexpr(bidirectional_iterator<Iter>)
    {
        cout<<"处理 bidirectional_iterator"<<endl;
    }
    else 
    {
        cout<<"其他迭代器"<<endl;
    }
}

// template<random_access_iterator Iter>
// void process (Iter start, Iter end )
// {
//     cout<<"处理 random_access_iterator"<<endl;
// }

// template<bidirectional_iterator Iter>
// void process (Iter start, Iter end )
// {
//     cout<<"处理 bidirectional_iterator"<<endl;
// }

// template<typename Iter>
// void process (Iter start, Iter end )
// {
//     cout<<"处理其他迭代器"<<endl;
// }
   

int main(){
    vector<int> v{1,2,3,4,5};
    process(v.begin(), v.end());

    map<string, int> m { {"C++", 100}, {"Rust", 200}, {"GO", 300}, };
    process(m.begin(), m.end());

    forward_list<int> l = { 7, 5, 16, 8 };
    process(l.begin(),l.end());

    int data[]={1,2,3,4,5};
    process(data, data+5);

}
    