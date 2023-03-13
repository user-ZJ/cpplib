#include <iostream>

import std;

import MyClass; // 导入模块

using namespace std;

int main()
{
    MyClass c{200};

    auto data=c.getData();

    cout<<data.value<<endl;

 


}