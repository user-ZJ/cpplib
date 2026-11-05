#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <array>

using namespace std;

constexpr bool isPrime (unsigned int p)
{
    for (unsigned int d=2; d<=p/2; ++d) {
        if (p % d == 0) {
            return false; 
        }
    }
    return p > 1; 
}

template<int Num>
consteval std::array<int, Num> primeNumbers()
{
    std::array<int, Num> primes;
    int idx = 0;
    for (int val = 1; idx < Num; ++val) {
    if (isPrime(val)) {
            primes[idx++] = val;
        }
    }
    return primes;
}


int main()
{
   


    constexpr int data=11;
    cout<<isPrime(data);

    auto primes1 = primeNumbers<data>();
    for (auto v : primes1) {
        std::cout << v << '\n';
    }
}

