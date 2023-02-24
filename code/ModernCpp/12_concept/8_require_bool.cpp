#include <iostream>
#include <vector>
#include <set>
#include <ranges>
#include <atomic>

using namespace std;


template<typename T>
requires (sizeof(T) > 4) // 编译时布尔值
&& requires { typename T::value_type; } // requires表达式
&& std::input_iterator<T> // 概念
void foreach(T begin, T end)
{

}

template<typename T>
requires std::integral<T> || std::floating_point<T>
T power(T b, T p)
{


}

template<typename T, typename U>
requires std::convertible_to<T, U>
auto f(T x, U y) {

}

template<typename T>
requires (sizeof(T) != sizeof(long))
auto F(T x){

}


template<typename T>
requires (sizeof(T) <= 64)
auto F(T x){

}

template<typename T, std::size_t Sz>
auto F(T x){

}

template<typename T>
requires (std::is_pointer_v<T> || std::same_as<T, std::nullptr_t>)
auto F(T x){

}

template<typename T>
requires (!std::convertible_to<T, std::string>)
auto F(T x){

}

//与string可转换
template<typename T>
requires (std::is_convertible_v<T, std::string>)
auto F(T x){

}


int main(){
    return 0;
}




