module;

#include <iostream>

export module Complex; // 声明模块Complex

using namespace std; 

//模块对外接口
export {

    class Complex{
    public:
        double re;
        double im;

        Complex(double r, double i):re(r), im(i){
        }

        void print() const;
    };

    Complex operator+(const Complex& c1, const Complex& c2);

}