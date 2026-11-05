
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <array>

using namespace std;



int main()
{
    auto hashed = [] (const char* str) consteval {
        std::size_t hash = 5381; 
        while (*str != '\0') {
            hash = hash * 33 ^ *str++;
        }
        return hash;
    };


    // 编译时上下文
    std::array arr{hashed("beer"), hashed("wine"), hashed("water")};

    cout<<arr[0]<<endl;
    cout<<arr[1]<<endl;
    cout<<arr[2]<<endl;
    
}