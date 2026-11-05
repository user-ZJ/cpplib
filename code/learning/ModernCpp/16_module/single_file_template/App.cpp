#include <iostream>

import Complex; // 导入模块



int main()
{
    Complex<int> c1{10,20};

    Complex c2{30,40};

    Complex<int> c3=c1+c2;

    c3.print();

}