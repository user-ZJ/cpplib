module;

#include <iostream>

// 声明模块Complex
export module MyClass; 




//模块对外接口
export class MyClass{
    MyData data;
public:
    MyClass(int value):data{value}{

    }

    MyData getData() const{
        return data;
    }

    
};

module :private;

struct MyData{
    int value;
};

 




