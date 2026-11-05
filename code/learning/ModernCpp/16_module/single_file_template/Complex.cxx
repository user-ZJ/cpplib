module; 

//全局模块段（#define, #include )

#include <iostream>

// 声明模块Complex
export module Complex; 

using namespace std; 


//模块对外接口
export template<typename T>
class Complex{
public:
    T re;
    T im;

    Complex(T r, T i):re(r), im(i){
    }

    void print() const
    {   
        cout<<"version 2.0"<<endl;
        cout<<re<<"+"<<im<<"i"<<endl;
    }


    
};

//模块对外接口
export template<typename T>
Complex<T> operator+(const Complex<T>& c1, const Complex<T>& c2)
{
    return Complex(c1.re+c2.re, c1.im+c2.im);
}





