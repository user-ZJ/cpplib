#include <iostream>
#include <any>
using namespace std;


template <typename T>
class Complex{
    T re, im;

public:



    T real() const {return re;}
    T imag() const {return im;}

    Complex(T _re, T _im):re{_re}, im{_im}
    {}

    Complex(const Complex& rhs):re{rhs.re}, im(rhs.im)
    {
        cout<<"copy ctor"<<endl;
    }

    template<typename U>
    Complex(const Complex<U>& c):
        re{static_cast<T>(c.real())},
        im{static_cast<T>(c.imag())}
        {
            cout<<"memeber template"<<endl;
        }
    

    template<typename U>
    void add(U r, U i) // 成员模板不可以是虚函数
    {
        this->re+=r;
        this->im+=i;
    }

   
    virtual void process(std::any r, std::any i) // 成员模板不可以是虚函数
    {
      
    }

     virtual void print(){
        cout<<"["<<re<<","<<im<<"]"<<endl;
    }

};

int main(){

    

    Complex<int> c1{100,200};
    
    Complex<int> c3=c1;


    Complex<double> c2{34.7, 89.9};
    Complex<int> c4=c2;

    Complex c5{1000,2000};
    pair p1{100,200};

    c2.add(100,200);
    c2.add(100.234,200.234);


}