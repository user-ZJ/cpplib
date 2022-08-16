# Eigen3使用笔记

官方文档：https://eigen.tuxfamily.org/dox/

Eigen是一个头文件依赖的库，不需要编译出库，直接包含头文件使用

## 使用示例
```cpp
#include <iostream>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
```
```shell
g++ -I /path/to/eigen/ my_program.cpp -o my_program 
```

## Matrix/Vector
**Eigen数据存储默认为列优先方式。**           
Eigen中矩阵和向量都是Matrix对象，vector是行或列为1的Matrix。  
Eigen中矩阵和向量分为固定维度和动态维度；固定维度的维度在编译时确定，运行过程中不能修改维度，内存在初始化时分配（在栈上）；动态维度的维度在运行时确定，运行时可以修改维度，内存分配在堆上。
通常Matrix的大小小于16时建议使用固定维度的矩阵，否则建议使用动态维度的矩阵。

**构造函数模板**      
```cpp
Matrix<typename Scalar,
       int RowsAtCompileTime,
       int ColsAtCompileTime,
       int Options = 0,   // RowMajor表示使用行优先存储
       int MaxRowsAtCompileTime = RowsAtCompileTime,  
       int MaxColsAtCompileTime = ColsAtCompileTime>
// MaxRowsAtCompileTime 和 MaxColsAtCompileTime设置静态的最大行数和最大列数，避免动态申请内存
// Matrix<float, Dynamic, Dynamic, 0, 3, 4> a;  // 静态分配3x4大小的内存，根据Dynamic大小，使用部分数据
```

**vector**           
Eigen中向量通常为列向量
```cpp
typedef Matrix<float, 3, 1> Vector3f;  // 3行1列的列向量，固定维度
typedef Matrix<int, 1, 2> RowVector2i;  // 1行2列的行向量，固定维度
typedef Matrix<float, Dynamic, 1> VectorXf; // 动态维度
// 初始化
Vector3f a;
Vector2d a(5.0, 6.0);
Vector3d b(5.0, 6.0, 7.0);
Vector4d c(5.0, 6.0, 7.0, 8.0);
VectorXf b(30);
VectorXf c{1, 2, 3, 4, 5};
VectorXf c={1, 2, 3, 4, 5};
VectorXd a {{1.5, 2.5, 3.5}};             // A column-vector with 3 coefficients
RowVectorXd b {{1.0, 2.0, 3.0, 4.0}};     // A row-vector with 4 coefficients
```

**Matrix**           
Eigen中矩阵通常为列优先
```cpp
typedef Matrix<float, 4, 4> Matrix4f;
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;  // 动态维度
// 初始化
Matrix4f a;
MatrixXd b(10,15);
Matrix<double, 2, 3> b {
      {2, 3, 4},
      {5, 6, 7},
};
// 逗号初始化
Matrix3f m;
m << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;
std::cout << m;
```

**resize/conservativeResize**
动态维度的Matrix可以进行resize，当resize前后Matrix的中元素个数不变，resize只改变shape大小，元素内容不变；否则元素内容会被设置成未初始化状态。  
conservativeResize同样是对动态维度的Matrix进行resize，但当Matrix中元素个数发生变化时，会保留原有数据，元素数据减少时，使用原有元素的部分数据；元素个数增加时，新增的数据会被设置为未初始化状态。
```cpp
#include <iostream>
#include <Eigen/Dense>
 
int main()
{
  Eigen::MatrixXd m(2,5);
  m.resize(4,3);
  std::cout << "The matrix m is of size "
            << m.rows() << "x" << m.cols() << std::endl;
  std::cout << "It has " << m.size() << " coefficients" << std::endl;
  Eigen::VectorXd v(2);
  v.resize(5);
  std::cout << "The vector v is of size " << v.size() << std::endl;
  std::cout << "As a matrix, v is of size "
            << v.rows() << "x" << v.cols() << std::endl;
}
```

**赋值构造**
赋值构造函数会先对左边的Matrix进行resize，再进行数据拷贝。

**常用的typedef**   
MatrixNt for Matrix<type, N, N>. For example, `MatrixXi` for Matrix<int, Dynamic, Dynamic>.      
MatrixXNt for Matrix<type, Dynamic, N>. For example, `MatrixX3i` for Matrix<int, Dynamic, 3>.              
MatrixNXt for Matrix<type, N, Dynamic>. For example, `Matrix4Xd` for Matrix<d, 4, Dynamic>.           
VectorNt for Matrix<type, N, 1>. For example, `Vector2f` for Matrix<float, 2, 1>.           
RowVectorNt for Matrix<type, 1, N>. For example, `RowVector3d` for Matrix<double, 1, 3>.            

N can be any one of 2, 3, 4, or X (meaning Dynamic).         
t can be any one of i (meaning int), f (meaning float), d (meaning double), cf (meaning complex<float>), or cd (meaning complex<double>).        

### 矩阵运算

**加减**            
Matrix重载了+，-，+=，-=
```cpp
#include <iostream>
#include <Eigen/Dense>
 
int main()
{
  Eigen::Matrix2d a;
  a << 1, 2,
       3, 4;
  Eigen::MatrixXd b(2,2);
  b << 2, 3,
       1, 4;
  std::cout << "a + b =\n" << a + b << std::endl;
  std::cout << "a - b =\n" << a - b << std::endl;
  std::cout << "Doing a += b;" << std::endl;
  a += b;
  std::cout << "Now a =\n" << a << std::endl;
  Eigen::Vector3d v(1,2,3);
  Eigen::Vector3d w(1,0,0);
  std::cout << "-v + w - v =\n" << -v + w - v << std::endl;
}
```

**缩放**             
Matrix重载了*，/，*=，/=
```cpp
#include <iostream>
#include <Eigen/Dense>
 
int main()
{
  Eigen::Matrix2d a;
  a << 1, 2,
       3, 4;
  Eigen::Vector3d v(1,2,3);
  std::cout << "a * 2.5 =\n" << a * 2.5 << std::endl;
  std::cout << "0.1 * v =\n" << 0.1 * v << std::endl;
  std::cout << "Doing v *= 2;" << std::endl;
  v *= 2;
  std::cout << "Now v =\n" << v << std::endl;
}
```

**转置**           
对于实数，conjugate不做任何操作，adjoint等于transpose;
对于虚数，conjugate对虚部取反，adjoint对transpose后的虚部取反
```cpp
MatrixXcf a = MatrixXcf::Random(2,2);
cout << "Here is the matrix a\n" << a << endl;
cout << "Here is the matrix a^T\n" << a.transpose() << endl;
cout << "Here is the conjugate of a\n" << a.conjugate() << endl;
cout << "Here is the matrix a^*\n" << a.adjoint() << endl;

MatrixXf a(2,3); a << 1, 2, 3, 4, 5, 6;
cout << "Here is the initial matrix a:\n" << a << endl;
a.transposeInPlace();  // 不能使用a=a.transpose();
cout << "and after being transposed:\n" << a << endl;
```

**矩阵/向量乘法**             
```cpp
#include <iostream>
#include <Eigen/Dense>
 
int main()
{
  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  Eigen::Vector2d u(-1,1), v(2,0);
  std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
  std::cout << "Here is mat*u:\n" << mat*u << std::endl;
  std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
  std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
  std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
  std::cout << "Let's multiply mat by itself" << std::endl;
  mat = mat*mat;
  std::cout << "Now mat is mat:\n" << mat << std::endl;
}
```

**点积、叉积**           
叉积又叫外积，在数学和向量代数领域，外积（英语：Cross product）又称向量积（英语：Vector product），是对**三维空间**中的两个向量的二元运算，使用符号 X。与点积不同，它的运算结果是向量。对于线性无关的两个向量 a和 b ，它们的外积写作 a X b，是 a 和 b  所在平面的法线向量，与 a  和 b 都垂直
```cpp
#include <iostream>
#include <Eigen/Dense>
 
int main()
{
  Eigen::Vector3d v(1,2,3);
  Eigen::Vector3d w(0,1,2);
 
  std::cout << "Dot product: " << v.dot(w) << std::endl;
  double dp = v.adjoint()*w; // automatic conversion of the inner product to a scalar
  std::cout << "Dot product via a matrix product: " << dp << std::endl;
  std::cout << "Cross product:\n" << v.cross(w) << std::endl;
}
```

**统计运算**         
```cpp
#include <iostream>
#include <Eigen/Dense>
 
using namespace std;
int main()
{
  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  cout << "Here is mat.sum():       " << mat.sum()       << endl;  // 所有元素的和
  cout << "Here is mat.prod():      " << mat.prod()      << endl;  // 所有元素的积
  cout << "Here is mat.mean():      " << mat.mean()      << endl;  // 均值
  cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << endl;  // 最小值
  cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << endl;  // 最大值
  cout << "Here is mat.trace():     " << mat.trace()     << endl;  // 对角线的和

  Matrix3f m = Matrix3f::Random();
  std::ptrdiff_t i, j;
  float minOfM = m.minCoeff(&i,&j);
  cout << "Here is the matrix m:\n" << m << endl;
  cout << "Its minimum coefficient (" << minOfM 
       << ") is at position (" << i << "," << j << ")\n\n";
 
  RowVector4i v = RowVector4i::Random();
  int maxOfV = v.maxCoeff(&i);
  cout << "Here is the vector v: " << v << endl;
  cout << "Its maximum coefficient (" << maxOfV 
       << ") is at position " << i << endl;
}
```

## Array
Eigen中Matrix提供提供线性代数相关运算，Array提供更通用的运算，如：给每个元素加上一个常量，将两个数组相乘  

```cpp
typedef Array<float,Dynamic,1> ArrayXf;
typedef Array<float,3,1> Array3f;
typedef Array<double,Dynamic,Dynamic> ArrayXXd;
typedef Array<double,3,3> Array33d;
```

```cpp
#include <Eigen/Dense>
#include <iostream>
 
int main()
{
  Eigen::ArrayXXf  m(2,2);
  // assign some values coefficient by coefficient
  m(0,0) = 1.0; m(0,1) = 2.0;
  m(1,0) = 3.0; m(1,1) = m(0,1) + m(1,0);
  // print values to standard output
  std::cout << m << std::endl << std::endl;
  // using the comma-initializer is also allowed
  m << 1.0,2.0,
       3.0,4.0;  
  // print values to standard output
  std::cout << m << std::endl;
}
```

### Array运算
**加减**   
```cpp
#include <Eigen/Dense>
#include <iostream>
 
int main()
{
  Eigen::ArrayXXf a(3,3);
  Eigen::ArrayXXf b(3,3);
  a << 1,2,3,
       4,5,6,
       7,8,9;
  b << 1,2,3,
       1,2,3,
       1,2,3;  
  // Adding two arrays
  std::cout << "a + b = " << std::endl << a + b << std::endl << std::endl;
  // Subtracting a scalar from an array
  std::cout << "a - 2 = " << std::endl << a - 2 << std::endl;
}
```

**乘法**：对应位置元素相乘    
```cpp
#include <Eigen/Dense>
#include <iostream>
 
int main()
{
  Eigen::ArrayXXf a(2,2);
  Eigen::ArrayXXf b(2,2);
  a << 1,2,
       3,4;
  b << 5,6,
       7,8;
  std::cout << "a * b = " << std::endl << a * b << std::endl;
}
```

**系数运算**：abs,sqrt,min   
```cpp
#include <Eigen/Dense>
#include <iostream>
 
int main()
{
  Eigen::ArrayXf a = Eigen::ArrayXf::Random(5);
  a *= 2;
  std::cout << "a =" << std::endl
            << a << std::endl;    // 绝对值
  std::cout << "a.abs() =" << std::endl
            << a.abs() << std::endl;
  std::cout << "a.abs().sqrt() =" << std::endl
            << a.abs().sqrt() << std::endl;  // 平方
  std::cout << "a.min(a.abs().sqrt()) =" << std::endl
            << a.min(a.abs().sqrt()) << std::endl;  // 两个Array中的最小值
}
```

### Array和Matrix之间转换
Matrix使用`.array()`转换为Array;Array使用`.matrix()`转换为Matrix。  
**注意：Eigen不允许Matrix和Array的混合运算。**   
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using Eigen::MatrixXf;
 
int main()
{
  MatrixXf m(2,2);
  MatrixXf n(2,2);
  MatrixXf result(2,2);
 
  m << 1,2,
       3,4;
  n << 5,6,
       7,8;
 
  result = m * n;
  std::cout << "-- Matrix m*n: --\n" << result << "\n\n";
  result = m.array() * n.array();
  std::cout << "-- Array m*n: --\n" << result << "\n\n";
  result = m.cwiseProduct(n);
  std::cout << "-- With cwiseProduct: --\n" << result << "\n\n";
  result = m.array() + 4;
  std::cout << "-- Array m + 4: --\n" << result << "\n\n";
}
```

## 块操作
块是指Matrix或Array中的一个矩形区域，使用块操作不会带来额外的时间开销   
```cpp
matrix.block(i,j,p,q);  // 动态维度，块大小为pxq,起始位置为(i,j)
matrix.block<p,q>(i,j);  // 固定维度，块大小为pxq,起始位置为(i,j)
```
两个版本的block都可以用于固定维度和动态维度的Matrix和Array   
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  Eigen::MatrixXf m(4,4);
  m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;
  cout << "Block in the middle" << endl;
  cout << m.block<2,2>(1,1) << endl << endl;
  for (int i = 1; i <= 3; ++i)
  {
    cout << "Block of size " << i << "x" << i << endl;
    cout << m.block(0,0,i,i) << endl << endl;
  }

  Eigen::Array22f m;
  m << 1,2,
       3,4;
  Eigen::Array44f a = Eigen::Array44f::Constant(0.6);
  std::cout << "Here is the array a:\n" << a << "\n\n";
  a.block<2,2>(1,1) = m;
  std::cout << "Here is now a with m copied into its central 2x2 block:\n" << a << "\n\n";
  a.block(0,0,2,3) = a.block(2,1,2,3);
  std::cout << "Here is now a with bottom-right 2x3 block copied into top-left 2x3 block:\n" << a << "\n\n";
}
```

### 行与列
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  Eigen::MatrixXf m(3,3);
  m << 1,2,3,
       4,5,6,
       7,8,9;
  cout << "Here is the matrix m:" << endl << m << endl;
  cout << "2nd Row: " << m.row(1) << endl;
  m.col(2) += 3 * m.col(0);
  cout << "After adding 3 times the first column into the third column, the matrix m is:\n";
  cout << m << endl;
}
```
### vector中应用block
```cpp
vector.head(n);  
vector.head<n>();
vector.tail(n);
vector.tail<n>();
vector.segment(i,n);
vector.segment<n>(i);
```

```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  Eigen::ArrayXf v(6);
  v << 1, 2, 3, 4, 5, 6;
  cout << "v.head(3) =" << endl << v.head(3) << endl << endl;
  cout << "v.tail<3>() = " << endl << v.tail<3>() << endl << endl;
  v.segment(1,4) *= 2;
  cout << "after 'v.segment(1,4) *= 2', v =" << endl << v << endl;
}
```

## Matrix和Array初始化
### comma initializer
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  Matrix3f m;
  m << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;
  std::cout << m;

  RowVectorXd vec1(3);
  vec1 << 1, 2, 3;
  std::cout << "vec1 = " << vec1 << std::endl;
 
  RowVectorXd vec2(4);
  vec2 << 1, 4, 9, 16;
  std::cout << "vec2 = " << vec2 << std::endl;
 
  RowVectorXd joined(7);
  joined << vec1, vec2;
  std::cout << "joined = " << joined << std::endl;

  MatrixXf matA(2, 2);
  matA << 1, 2, 3, 4;
  MatrixXf matB(4, 4);
  matB << matA, matA/10, matA/10, matA;
  std::cout << matB << std::endl;

  Matrix3f m;
  m.row(0) << 1, 2, 3;
  m.block(1,0,2,2) << 4, 5, 7, 8;
  m.col(2).tail(2) << 6, 9;                   
  std::cout << m;
}
```

### Zero
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  std::cout << "A fixed-size array:\n";
  Array33f a1 = Array33f::Zero();  // 不带参数的Zero只能用于固定维度的Matrix/Array初始化
  std::cout << a1 << "\n\n";
 
 
  std::cout << "A one-dimensional dynamic-size array:\n";
  ArrayXf a2 = ArrayXf::Zero(3);
  std::cout << a2 << "\n\n";
 
 
  std::cout << "A two-dimensional dynamic-size array:\n";
  ArrayXXf a3 = ArrayXXf::Zero(3, 4);   // 等价于a3.setZero(3, 4);
  std::cout << a3 << "\n";
}
```

### Constant
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  std::cout << "A fixed-size array:\n";
  Eigen::Array33f a1 = Eigen::Array33f::Constant(0.5);  // 不带参数的Constant只能用于固定维度的Matrix/Array初始化
  std::cout << a1 << "\n\n";
 
 
  std::cout << "A one-dimensional dynamic-size array:\n";
  Eigen::ArrayXf a2 = Eigen::ArrayXf::Constant(3,0.5);
  std::cout << a2 << "\n\n";
 
 
  std::cout << "A two-dimensional dynamic-size array:\n";
  Eigen::ArrayXXf a3 = Eigen::ArrayXXf::Constant(3, 4,0.5);  // 等价于a3.setConstant(3, 4,0.5);
  std::cout << a3 << "\n";
}
```

### Random
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  std::cout << "A fixed-size array:\n";
  Eigen::Array33f a1 = Eigen::Array33f::Random();  // 不带参数的Random只能用于固定维度的Matrix/Array初始化
  std::cout << a1 << "\n\n";
 
 
  std::cout << "A one-dimensional dynamic-size array:\n";
  Eigen::ArrayXf a2 = Eigen::ArrayXf::Random(3);
  std::cout << a2 << "\n\n";
 
 
  std::cout << "A two-dimensional dynamic-size array:\n";
  Eigen::ArrayXXf a3 = Eigen::ArrayXXf::Random(3, 4);  // 等价于a3.setRandom(3, 4);
  std::cout << a3 << "\n";
}
```

### Identity
将对角线设置为1（行id等于列id），其余元素设置为0  
Identity只能用于Matrix，不能用于Array，因为Identity操作是线性代数的操作。
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  std::cout << "A fixed-size array:\n";
  Eigen::Matrix3f a1 = Eigen::Matrix3f::Identity();  // 不带参数的Random只能用于固定维度的Matrix/Array初始化
  std::cout << a1 << "\n\n";
 
 
  std::cout << "A two-dimensional dynamic-size array:\n";
  Eigen::MatrixXf a3 = Eigen::MatrixXf::Identity(3, 4); // 等价于a3.setIdentity(3, 4);
  std::cout << a3 << "\n";
}
```

### LinSpaced
LinSpaced只适用于vector或者以为的Array，生成间距相等的数据
```cpp
#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
 
int main()
{
  Eigen::ArrayXXf table(10, 4);
  table.col(0) = Eigen::ArrayXf::LinSpaced(10, 0, 90);  //等价于table.col(0).setLinSpaced(10, 0, 90);
  table.col(1) = M_PI / 180 * table.col(0);
  table.col(2) = table.col(1).sin();
  table.col(3) = table.col(1).cos();
  std::cout << "  Degrees   Radians      Sine    Cosine\n";
  std::cout << table << std::endl;
}
```

## 导入raw数据到Eigen
Eigen适用Map映射内容到Matrix 

**构造函数**  
Map构造函数需要传入数据的首地址，指定的shape   
```cpp
Map<MatrixXf> mf(pf,rows,columns);  // 动态维度matrix Map，需要传入shape
Map<const Vector4i> mi(pi);   // 静态维度的matrix Map，不需要传入shape
// 其他模板参数
Map<typename MatrixType,
    int MapOptions,   // 指定内存是否对齐
    typename StrideType>  // 指定各维度的stride
```

```cpp
// MapOptions和StrideType使用实例
#include <Eigen/Dense>
#include <iostream>

int main() {
  int array[8];
  for (int i = 0; i < 8; ++i) array[i] = i;
  std::cout << "Column-major:\n" << Eigen::Map<Eigen::Matrix<int, 2, 4>>(array) << std::endl;
  std::cout << "Row-major:\n" << Eigen::Map<Eigen::Matrix<int, 2, 4, Eigen::RowMajor>>(array) << std::endl;
  std::cout << "Row-major using stride:\n" << Eigen::Map<Eigen::Matrix<int, 2, 4>, Eigen::Unaligned, Eigen::Stride<1, 4>>(array) << std::endl;
}
```

```cpp
// Map变量使用示例
#include <Eigen/Dense>
#include <iostream>

int main() {
  typedef Eigen::Matrix<float, 1, Eigen::Dynamic> MatrixType;
  typedef Eigen::Map<MatrixType> MapType;
  typedef Eigen::Map<const MatrixType> MapTypeConst;  // a read-only map
  const int n_dims = 5;

  MatrixType m1(n_dims), m2(n_dims);
  m1.setRandom();
  m2.setRandom();
  float *p = &m2(0);                      // get the address storing the data for m2
  MapType m2map(p, m2.size());            // m2map shares data with m2
  MapTypeConst m2mapconst(p, m2.size());  // a read-only accessor for m2

  std::cout << "m1: " << m1 << std::endl;
  std::cout << "m2: " << m2 << std::endl;
  std::cout << "Squared euclidean distance: " << (m1 - m2).squaredNorm() << std::endl;
  std::cout << "Squared euclidean distance, using map: " << (m1 - m2map).squaredNorm() << std::endl;
  m2map(3) = 7;  // this will change m2, since they share the same array
  std::cout << "Updated m2: " << m2 << std::endl;
  std::cout << "m2 coefficient 2, constant accessor: " << m2mapconst(2) << std::endl;
  /* m2mapconst(2) = 5; */  // this yields a compile-time error
}
```

```cpp
// 修改Map对象示例
#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

int main() {
  // 修改Map变量，映射到新地址
  int data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  Eigen::Map<Eigen::RowVectorXi> v(data, 4);
  std::cout << "The mapped vector v is: " << v << "\n";
  // 注意：这里只是使用了new关键字，并不会申请内存，只是重新映射了内存地址
  new (&v) Eigen::Map<Eigen::RowVectorXi>(data + 4, 5);
  std::cout << "Now v is: " << v << "\n";

  // 使用空地址初始化map对象，在后续使用中再重新修改map对象
  float fdata[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  Eigen::Map<Eigen::Matrix3f> A(NULL);  // don't try to use this matrix yet!
  Eigen::VectorXf b(3);
  for (int i = 0; i < 3; i++) {
    new (&A) Eigen::Map<Eigen::Matrix3f>(&fdata[i]);
    b(i) = A.trace();
  }
  std::cout << b << std::endl;
}
```
