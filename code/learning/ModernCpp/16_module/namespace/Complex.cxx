module;

#include <iostream>

// 声明模块Complex
export module Math; 


//模块对外接口
export namespace Math{

    class Complex{
    public:
        double re;
        double im;

        Complex(double r, double i):re(r), im(i){
        }

        void print() const
        {
            std::cout<<re<<"+"<<im<<"i"<<std::endl;
        }

        static void process(){
            std::cout<<"process"<<std::endl;
        }
        
    };

    //模块对外接口
    Complex operator+(const Complex& c1, const Complex& c2)
    {
        return Complex(c1.re+c2.re, c1.im+c2.im);
    }

}
    




