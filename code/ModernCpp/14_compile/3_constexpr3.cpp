#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <forward_list>

using namespace std;

//可支持编译器、也可支持运行时
constexpr bool isPrime (unsigned int p)
{
    //cout<<"hello"<<endl;

    for (unsigned int d=2; d<=p/2; ++d) {
        if (p % d == 0) {
            return false; 
        }
    }


    return p > 1; 
}

using CCharArray=const char[6];

constexpr int getLen (const char* text)
{
    int i=0;

    while(text[i]!='\0')
    {
        i++;
    }

    return i;
}
  
// constexpr int getLen2 (string text)
// {
//     int i=0;

//     while(text[i]!='\0')
//     {
//         i++;
//     }

//     return i;
// }
  

constexpr int getLen3 (string_view text)
{

    return text.length();
}
  





int main()
{
    int data=100;
    bool b1=isPrime(data);
    constexpr bool b2=isPrime(197);

    //cout<<b2<<endl;
    cout<<b1<<endl;
    
    // bool b3=isPrime(197);
    // cout<<b3<<endl;

    constexpr const char str[] = "Hello, world!";
    constexpr const char* s="helloworld";

    constexpr int len=getLen(str);
    cout<<len<<endl;

    // constexpr auto s2="hello";
    // constexpr int len2=getLen2(s2);
    // cout<<len2<<endl;

    constexpr std::string_view my_string = "Hello, world!";
    constexpr int len3=getLen3(my_string);
    cout<<len3<<endl;

  

}

