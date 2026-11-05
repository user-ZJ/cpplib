#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <forward_list>

using namespace std;

/*
void print1 ()
{
    cout<<endl;
}

template<typename T, typename... TT>
void print1 (T firstArg, TT... args)
{
  cout << firstArg << ' ';  
  print1(args...);          
}

void print1 (int , string, double, bool);
void print1 (string, double, bool);
void print1 (double, bool);
void print1 (bool);
*/




template<typename T, typename... Types>
void print (T const& firstArg, Types const&... args)
{
    cout << firstArg << endl;
    if constexpr(sizeof...(args) > 0) {
        print(args...); 
    }
}

// void print (int , string s, double d, bool b)
// {
//     cout << firstArg << endl;
//     print (s, d, b);
// }

// void print (string, double d, bool b)
// {
//     cout << firstArg << endl;
//     print (d, b);
// }
// void print (double, bool b)
// {
//     cout << firstArg << endl;
//     print (b);

// }
// void print (bool)
// {
//     cout << firstArg << endl;
// }


int main(){
    
    print(100,"hello"s, 30.46, true);

}
    