module;

#include <iostream>

module Complex; //模块实现单元

using namespace std; 



Complex operator+(const Complex& c1, const Complex& c2)
{
    return Complex(c1.re+c2.re, c1.im+c2.im);
}



