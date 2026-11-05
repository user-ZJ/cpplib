#include <iostream>
#include <vector>

using namespace std;

struct Complex{
    int re;
    int im;

    Complex(int _re=0, int _im=0):re(_re),im(_im){

    }
};

Complex operator+(const Complex& r1, const Complex& r2)
{
    return Complex(r1.re+r2.re, r1.im+r2.im);
}


//traits template
template<typename T>
struct SumTraits{
    using SumT = T;

};

//特化版本
template<>
struct SumTraits<char> {
    using SumT = int;
};

template<>
struct SumTraits<short> {
    using SumT = int;
};

template<>
struct SumTraits<int> {
    using SumT = long;
};

template<>
struct SumTraits<unsigned int> {
    using SumT = unsigned long;
};

template<>
struct SumTraits<float> {
    using SumT = double;
};



// 模板函数
template<typename T>
T sum0 (T const* beg, T const* end)
{

    T total{};  
    while (beg != end) {
        total += *beg;
        ++beg;
    }
    return total;
}



template<typename T>
auto sum1 (T const* beg, T const* end)
{
    using SumType = typename SumTraits<T>::SumT;

    SumType total{};  
    while (beg != end) {
        total += *beg;
        ++beg;
    }
    return total;
}



template<typename T, typename ST=SumTraits<T>>
auto sum2 (T const* beg, T const* end)
{

    typename ST::SumT total{};  
    while (beg != end) {
        total += *beg;
        ++beg;
    }
    return total;
}

template<typename Iter>
auto sum3 (Iter start, Iter end)
{
    using VT = typename std::iterator_traits<Iter>::value_type;

    using SumType = typename SumTraits<VT>::SumT;

    SumType total{};  
    while (start != end) {
        total =total+ *start;
        ++start;
    }
    return total;
}

int main()
{

    int num[] = { 1, 2, 3, 4, 5 };
    char name[] = "templates";
    vector<int> vec{10,20,30,40,50};

     int s0=sum0(num,num+5);
     int s1=sum0(name,name+sizeof(name)-1);
    
    
     cout<<s0<<","<<s1<<endl;



    int s2= sum1(num, num+5);
    int s3= sum1(name, name+sizeof(name)-1);
    cout<<s2<<","<<s3<<endl;



    int s4=sum2(num, num+5);
    int s5=sum2(name, name+sizeof(name)-1);
    cout<<s4<<","<<s5<<endl;

    int s6=sum3(num, num+5);
    int s7=sum3(name, name+sizeof(name)-1);
    int s8=sum3(vec.begin(), vec.end());
    cout<<s6<<","<<s7<<","<<s8<<endl;

    
    vector<Complex> vc{ Complex{1,2}, Complex{3,4}, Complex{5,6}};

    Complex cp=sum3(vc.begin(), vc.end());
    
    cout<<cp.re<<"+" <<cp.im<<"i"<<endl;

}


