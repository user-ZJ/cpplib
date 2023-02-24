
#include <iostream>
#include <array>

// 运行时
int squareR(int x) {
  return x * x;
}

// 编译时
consteval int squareC(int x) {
  return x * x;
}

// 编译时、运行时
constexpr int squareCR(int x) {
  return x * x;
}

int main()
{
  int data=10;
  int i = data*2;
  constinit static int ci=42;
  constexpr int ce=42;

  std::cout << squareR(i) << '\n';      // OK
  std::cout << squareCR(i) << '\n';     // OK 
  //std::cout << squareC(i) << '\n';       // ERROR

  ci++;

  std::cout << squareR(ci) << '\n';      // OK
  std::cout << squareCR(ci) << '\n';     // OK 
  //std::cout << squareC(ci) << '\n';       // ERROR


  std::cout << squareR(ce) << '\n';     // OK
  std::cout << squareCR(ce) << '\n';    // OK 
  std::cout << squareC(ce) << '\n';     // OK: 


  //std::array<int, squareR(42)> arr1;   // ERROR
  std::array<int, squareCR(42)> arr2;   // OK: 
  std::array<int, squareC(42)> arr3;    // OK: 
  //std::array<int, squareC(i)> arr4;    // ERROR
}


