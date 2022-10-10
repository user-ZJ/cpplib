blas库
==================

BLAS(Basic Linear Algebra Subroutines 基础线性代数程序集)是一个应用程序接口（API）标准。用以规范发布基础线性代数操作的数值库（如矢量或矩阵乘法）。

包括BLAS Level 1(向量-向量运算)、Level 2(向量-矩阵运算)和Level 3(矩阵-矩阵运算)。

BLAS只是一个接口(规范)，有很多不同的实现。最早的[BLAS的参考实现](http://www.netlib.org/blas/)可以追溯到1979年，这是由netlib维护的。
参考实现没有任何优化，只是一个正确的实现，它的目的是用于验证其它实现的正确性。MKL、ATLAS和OpenBLAS提供了优化的BLAS实现。



Linear Algebra PACKage (LAPACK)是一些线性代数的routine，最早使用Fortran实现。它包括比BLAS更高级的routine，比如矩阵求逆、SVD分解等等。
[LAPACK的参考实现](https://github.com/Reference-LAPACK)也是由Netlib维护。LAPACK内部会使用BLAS。
当然也可以混合使用的实现(比如使用ATLAS的BLAS来实现Netlib的LAPACK)

ACML（AMD Core Math Library）:厂商AMD的BLAS实现。

ATLAS: BSD许可证开源的BLAS实现。

CBLAS:是C语言版本的BLAS接口。

CLAPACK是使用f2c工具自动把Fortran转换成C版本的代码

MKL完全用C语言由自己来实现BLAS和LAPACK；MKL提供了一个非常高度优化的线性代数函数的实现，尤其是对Intel的CPU。事实上，这个库包括多种代码路径，它会根据CPU的每种不同特性选择最优的代码路径。因此在MKL里，不需要任何手动配置，它会自动的使用所有CPU的特性和特殊指令集(比如AVX2和AVX512)

Automatically Tuned Linear Algebra Software (ATLAS)是一个常用的BLAS实现以及LAPACK的**部分实现**。ATLAS的基本想法是根据不同的处理器来进行自动的调整，因此它的编译过程非常复杂和费时。因此要编译ATLAS会非常tricky。ATLAS通常比Netlib的参考实现的BLAS要高效。但是ATLAS值包含部分的LAPACK函数。它包括矩阵的求逆和Cholesky分解，但是没有SVD分解。因此kaldi实现了一些LAPACK函数(SVD和特征值分解)。

OpenBLAS实现BLAS和部分LAPACK

EIGEN是基于C++实现的可以用来进行线性代数、矩阵、向量操作等运算的库，采用源码的方式提供给用户，支持多平台。

blas API
-----------------------
https://netlib.org/blas/#_blas_routines

参考
----------------

http://fancyerii.github.io/kaldidoc/ext-matrix/

https://www.zhihu.com/question/27872849

https://zh.m.wikipedia.org/zh-hans/BLAS

https://zh.m.wikipedia.org/zh-hans/LAPACK