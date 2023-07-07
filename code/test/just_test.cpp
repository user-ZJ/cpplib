#include <iostream>
#include <random>

int main() {
  enum DataType { INT, FLOAT, DOUBLE, CHAR };
  static const int ARRAY_SIZE[DataType::CHAR + 1] = {
    [DataType::INT] = 4, [DataType::FLOAT] = 4, [DataType::DOUBLE] = 8, [DataType::CHAR] = 1};
  for (auto &v : ARRAY_SIZE)
    std::cout << v << " ";
  return 0;
}