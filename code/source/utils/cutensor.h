/*
 * @Author: zack
 * @Date: 2021-10-05 10:26:48
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 15:21:49
 */
#ifndef BASE_CUTENSOR_UTIL_H_
#define BASE_CUTENSOR_UTIL_H_
#include "utils/ctensor.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace BASE_NAMESPACE {

template <typename T>
struct CudaMemDeleter {
  void operator()(T *p) noexcept {
    cudaFree(p);
  }
};

// cutensor始终有数据的所有权,且数据始终是连续的
template <typename T>
class CuTensor {
 public:
  CuTensor();
  // 支持vector初始化
  explicit CuTensor(const std::vector<int> &shapes);
  // 支持initializer_list初始化
  explicit CuTensor(const std::initializer_list<int> &shapes);
  // 拷贝构造函数
  CuTensor(const CuTensor &t);
  // 移动构造函数
  CuTensor(CuTensor &&t) noexcept;
  // 赋值构造函数
  CuTensor &operator=(const CuTensor &t);
  // 移动赋值构造函数
  CuTensor &operator=(CuTensor &&t) noexcept;

  ~CuTensor();
  // 调整tensor大小，会重新分配内存，保证内存连续
  void resize(const std::vector<int> shapes);
  void resize(const std::initializer_list<int> &shapes);
  // 清除数据
  void clear();
  // 访问数据
  T *data();
  T *data() const;
  void fromcpu(const CTensor<T, int> &t);
  // 将数据拷贝到CPU内存
  CTensor<T, int> cpu();
  std::vector<int> shapes() const;
  std::vector<int> strides() const;
  uint64_t size() const;
  uint64_t byteSize() const;
  void create_tensor_desc();
  void distroy_tensor_desc();
  cudnnTensorDescriptor_t tensor_desc() const;

 private:
  std::shared_ptr<T> data_;
  T *ptr;
  std::vector<int> shapes_;
  std::vector<int> strides_;
  long size_;
  bool is_tensor_ = false;
  cudnnTensorDescriptor_t tensor_desc_;
};

template <typename T>
CuTensor<T>::CuTensor() {
  data_ = nullptr;
  size_ = 0;
  shapes_.clear();
  strides_.clear();
}

template <typename T>
CuTensor<T>::CuTensor(const std::vector<int> &shapes) {
  assert(shapes.size() > 0);
  shapes_ = shapes;
  strides_.resize(shapes.size());
  size_ = shapes_[0];
  strides_[shapes.size() - 1] = 1;
  for (int i = shapes_.size() - 1; i > 0; i--) {
    // assert(shapes_[i]>0);
    size_ *= shapes_[i];
    strides_[i - 1] = strides_[i] * shapes_[i];
  }
  cudaMalloc((void **)&ptr, sizeof(T) * size_);
  data_ = std::shared_ptr<T>(ptr, CudaMemDeleter<T>());
  create_tensor_desc();
}

// 支持initializer_list初始化
template <typename T>
CuTensor<T>::CuTensor(const std::initializer_list<int> &shapes) {
  assert(shapes.size() > 0);
  shapes_ = shapes;
  strides_.resize(shapes.size());
  size_ = shapes_[0];
  strides_[shapes.size() - 1] = 1;
  for (int i = shapes_.size() - 1; i > 0; i--) {
    // assert(shapes_[i]>0);
    size_ *= shapes_[i];
    strides_[i - 1] = strides_[i] * shapes_[i];
  }
  cudaMalloc((void **)&ptr, sizeof(T) * size_);
  data_ = std::shared_ptr<T>(ptr, CudaMemDeleter<T>());
  create_tensor_desc();
}

// 拷贝构造函数
template <typename T>
CuTensor<T>::CuTensor(const CuTensor<T> &t) {
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  cudaMalloc((void **)&ptr, sizeof(T) * size_);
  data_ = std::shared_ptr<T>(ptr, CudaMemDeleter<T>());
  cudaMemcpy(data_.get(), t.data_.get(), size_ * sizeof(T), cudaMemcpyDeviceToDevice);
  create_tensor_desc();
}

// 移动构造函数
template <typename T>
CuTensor<T>::CuTensor(CuTensor<T> &&t) noexcept {
  shapes_ = std::move(t.shapes_);
  strides_ = std::move(t.strides_);
  size_ = t.size_;
  data_ = t.data_;
  // t.data_ = nullptr;
  is_tensor_ = t.is_tensor_;
  tensor_desc_ = t.tensor_desc_;
  t.tensor_desc_ = nullptr;
  t.is_tensor_ = false;
}

// 赋值构造函数
template <typename T>
CuTensor<T> &CuTensor<T>::operator=(const CuTensor<T> &t) {
  if (this == &t) return *this;
  // copy & swap
  CuTensor temp{t};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(data_, temp.data_);
  std::swap(tensor_desc_, temp.tensor_desc_);
  std::swap(is_tensor_, temp.is_tensor_);
  return *this;
}

// 移动赋值构造函数
template <typename T>
CuTensor<T> &CuTensor<T>::operator=(CuTensor<T> &&t) noexcept {
  if (this == &t) return *this;
  // copy & swap
  CuTensor temp{std::move(t)};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(data_, temp.data_);
  std::swap(tensor_desc_, temp.tensor_desc_);
  std::swap(is_tensor_, temp.is_tensor_);
  return *this;
}

template <typename T>
CuTensor<T>::~CuTensor() {
  distroy_tensor_desc();
}

// 调整tensor大小，会重新分配内存，保证内存连续
template <typename T>
void CuTensor<T>::resize(const std::vector<int> shapes) {
  if (shapes == shapes_) return;
  CuTensor newTensor(shapes);
  std::swap(size_, newTensor.size_);
  std::swap(shapes_, newTensor.shapes_);
  std::swap(strides_, newTensor.strides_);
  std::swap(data_, newTensor.data_);
  std::swap(tensor_desc_, newTensor.tensor_desc_);
  std::swap(is_tensor_, newTensor.is_tensor_);
}

template <typename T>
void CuTensor<T>::resize(const std::initializer_list<int> &shapes) {
  std::vector<int> s{shapes};
  resize(s);
}

// 清除数据
template <typename T>
void CuTensor<T>::clear() {
  shapes_.clear();
  strides_.clear();
  size_ = 0;
}

template <typename T>
T *CuTensor<T>::data() {
  return data_.get();
}

template <typename T>
T *CuTensor<T>::data() const {
  return data_.get();
}

template <typename T>
std::vector<int> CuTensor<T>::shapes() const {
  return shapes_;
}

template <typename T>
std::vector<int> CuTensor<T>::strides() const {
  return strides_;
}

template <typename T>
uint64_t CuTensor<T>::size() const {
  return size_;
}
template <typename T>
uint64_t CuTensor<T>::byteSize() const {
  return size_ * sizeof(T);
}

template <typename T>
void CuTensor<T>::fromcpu(const CTensor<T, int> &t) {
  if (size_ != t.size() || shapes_.size() != t.shapes().size()) {
    shapes_ = t.shapes();
    strides_ = t.strides();
    size_ = t.size();
    cudaMalloc((void **)&ptr, sizeof(T) * size_);
    data_ = std::shared_ptr<T>(ptr, CudaMemDeleter<T>());
  }
  cudaMemcpy(data_.get(), t.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
CTensor<T, int> CuTensor<T>::cpu() {
  CTensor<T, int> t(shapes_);
  cudaMemcpy(t.data(), data_.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost);
  return t;
}

template <typename T>
void CuTensor<T>::distroy_tensor_desc() {
  if (is_tensor_ and tensor_desc_) {
    cudnnDestroyTensorDescriptor(tensor_desc_);
    is_tensor_ = false;
  }
}

template <typename T>
void CuTensor<T>::create_tensor_desc() {
  if (shapes_.size() <= 4) {
    std::array<int, 4> nchw{1, 1, 1, 1};
    for (int i = 0; i < shapes_.size(); i++)
      nchw[i] = shapes_[i];
    cudnnCreateTensorDescriptor(&tensor_desc_);
    cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, nchw[0], nchw[1], nchw[2], nchw[3]);
    is_tensor_ = true;
  }
}

template <typename T>
cudnnTensorDescriptor_t CuTensor<T>::tensor_desc() const {
  return tensor_desc_;
}

};  // namespace BASE_NAMESPACE

#endif