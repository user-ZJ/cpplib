#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <array>

using namespace std;

struct MyClass{
    constexpr bool isPrime (unsigned int p)
    {
        for (unsigned int d=2; d<=p/2; ++d) {
            if (p % d == 0) {
                return false; 
            }
        }
        return p > 1; 
    }
};


template<int Num>
constexpr std::array<int, Num> primeNumbers()
{
    MyClass*  mc=new MyClass();

    std::array<int, Num> primes;
    int idx = 0;
    for (int val = 1; idx < Num; ++val) {
    if (mc->isPrime(val)) {
            primes[idx++] = val;
        }
    }

    delete mc;

    return primes;
}


int main()
{
   

    constexpr array<int,10> primes1 = primeNumbers<10>();
    for (auto v : primes1) {
        std::cout << v << '\n';
    }

    
}

