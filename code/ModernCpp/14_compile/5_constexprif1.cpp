#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <forward_list>

using namespace std;

template <typename Iter>
using Iter_Category=typename iterator_traits<Iter>::iterator_category;



// template<typename Iter>
// void process_tag(Iter start, Iter end, forward_iterator_tag )
// {
//     cout<<"处理 forward_iterator"<<endl;
// }

// template<typename Iter>
// void process_tag(Iter start, Iter end, random_access_iterator_tag )
// {
//     cout<<"处理 random_access_iterator"<<endl;
// }

// template<typename Iter>
// void process_tag(Iter start, Iter end, bidirectional_iterator_tag )
// {
//     cout<<"处理 bidirectional_iterator"<<endl;
// }


// template<typename Iter>
// void process_tag(Iter start, Iter end, output_iterator_tag  )
// {
//     cout<<"处理 output_iterator"<<endl;
// }

// template<typename Iter>
// void process_tag(Iter start, Iter end, input_iterator_tag )
// {
//     cout<<"处理 input_iterator"<<endl;
// }

// template<typename Iter>
// void process (Iter start, Iter end )
// {
//     using iterator_category=typename iterator_traits<Iter>::iterator_category;
    
//     process_tag(start, end, iterator_category{});
// }




template<typename Iter>
void process (Iter start, Iter end )
{
    using iterator_category=typename iterator_traits<Iter>::iterator_category;
    
    //process_tag(start, end, iterator_category{});

    if constexpr( is_same<iterator_category,random_access_iterator_tag>::value)
    {
       cout<<"处理 random_access_iterator"<<endl;
    }
    else if constexpr( is_same<iterator_category,bidirectional_iterator_tag>::value)
    {
        cout<<"处理 bidirectional_iterator"<<endl;
    }
    else 
    {
        cout<<"其他迭代器"<<endl;
    }
}


// template<typename Iter>
// void process (Iter start, Iter end )
// {
//     using iterator_category=typename iterator_traits<Iter>::iterator_category;
//     using value_type=typename iterator_traits<Iter>::value_type;

//     if constexpr( is_same<iterator_category,random_access_iterator_tag>::value)
    
//         try{
       
            
//         if constexpr( is_pointer<value_type>::value)
        
//             }catch(invalid_argument e)
//             {

//             }
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
    