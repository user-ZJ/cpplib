#include <iostream>

template<typename T>
void process(T) {
  std::cout << "T is reference: " << std::is_reference_v<T> << '\n';
}

template<typename T>
void processR(T&) {
  std::cout << "T is reference: " << std::is_reference_v<T&> << '\n';
}

int main()
{
  std::cout << std::boolalpha;
  int i;
  int& r = i;
  process(i);         // false
  process(r);         // false
  process<int&>(i);   // true
  process<int&>(r);   // true

  processR(i);
  processR(r);
  processR<int&>(i);
  processR<int&>(r);
}
