#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <forward_list>

using namespace std;

class Complex{
public:
    double re;
    double im;

    constexpr Complex(double r, double i):re(r), im(i)
    {
        //cout<<"ctor"<<endl;
    }

    constexpr double get_re() const noexcept{
        return re;
    }

    constexpr double get_im() const noexcept{
        return im;
    }

    constexpr virtual void print() const{   //constexpr
        // cout<<re<<"+"<<im<<"i"<<endl;
    }

};

constexpr Complex operator+(const Complex& c1, const Complex& c2)
{

    return Complex(c1.re+c2.re, c1.im+c2.im);

}


int main()
{
    constexpr Complex c1{10.1,20.2};

    constexpr Complex c2{100.01,200.02};

    constexpr Complex c3=c1+c2;

    constexpr auto r=c3.get_re();
    constexpr auto i=c3.get_im();

    cout<<c3.get_re()<<","<<c3.get_im()<<endl;
    c3.print();
    

}

