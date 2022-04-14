#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

int main() {
  Eigen::MatrixXi a(2,2);
  a<<
    // construct a 2x2 matrix
    1, 2,  // first row
    3, 4   // second row
  ;
  Eigen::Matrix<double, 2, 3> b;
  b<<
    2, 3, 4,
    5, 6, 7;
  std::cout << b<< std::endl;
}