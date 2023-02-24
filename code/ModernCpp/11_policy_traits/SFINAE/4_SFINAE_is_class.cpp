#include <iostream>
#include <string>
#include <memory>
#include <vector>
using namespace std;


template<typename T>
class IsClassT {
private:
    typedef char One; //1 byte
    typedef struct { char a[2]; } Two; //2byte

    //模板函数不要求函数必须完整实现
    template<typename C> 
    static One test(int C::*); // C支持::运算符，较强绑定类型

    template<typename C> 
    static Two test(...); // C不支持支持::运算符，较弱绑定类型
public:
    //static constexpr bool Value = sizeof( IsClassT<T>::test<T>(0)) == 1;

    static constexpr bool Value = (sizeof( IsClassT<T>::test<T>(0)) == 1) && !is_union<T>::value;
};


struct Point
{
  
};

enum class Color{
    red, 
    blue,
    green,
};

enum CColor{
    red, 
    blue,
    green,
};

union BigInt{
    int data;
    long value;
};

int main()
{
    cout<<std::boolalpha;

    cout<<IsClassT<int>::Value<<endl;
    cout<<IsClassT<string>::Value<<endl;
    cout<<IsClassT<Point>::Value<<endl;
    cout<<IsClassT<Color>::Value<<endl;
    cout<<IsClassT<CColor>::Value<<endl;
    cout<<IsClassT<BigInt>::Value<<endl;

    cout<<endl;
    
    cout<<std::is_class<int>::value<<endl;
    cout<<std::is_class<string>::value<<endl;
    cout<<std::is_class_v<Point><<endl;
    cout<<std::is_class_v<Color><<endl;
    cout<<std::is_class_v<CColor><<endl;
    cout<<std::is_class_v<BigInt><<endl;




}