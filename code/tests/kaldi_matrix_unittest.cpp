#include <gtest/gtest.h>
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"

TEST(KaldiMatrix, BlasProductAndLapackSingularValues) {
  kaldi::Matrix<double> a(2, 2), product(2, 2);
  a(0, 0) = 3.0;
  a(1, 1) = 4.0;
  product.AddMatMat(1.0, a, kaldi::kNoTrans, a, kaldi::kTrans, 0.0);
  EXPECT_NEAR(product(0, 0), 9.0, 1e-10);
  EXPECT_NEAR(product(1, 1), 16.0, 1e-10);
  EXPECT_NEAR(product(0, 1), 0.0, 1e-10);
  kaldi::Vector<double> singular_values(2);
  a.Svd(&singular_values);
  double sum = singular_values(0) + singular_values(1);
  double squares = singular_values(0) * singular_values(0) +
                   singular_values(1) * singular_values(1);
  EXPECT_NEAR(sum, 7.0, 1e-10);
  EXPECT_NEAR(squares, 25.0, 1e-10);
}
