#include <iostream>

import Math; // 导入模块

using namespace Math;


int main()
{
    Complex c1{10.1,20.2};

    Complex c2{30.3,40.4};

    Complex c3=c1+c2;

    c3.print();

    Complex::process();
}