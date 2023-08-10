/*
 * @Author: zack
 * @Date: 2021-10-05 10:26:48
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 15:21:49
 */
#ifndef _BASE_CUTENSOR_UTIL_H_
#define _BASE_CUTENSOR_UTIL_H_
#include "ctensor.h"
#include "cuhelper.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace CUDA_NAMESPACE {

enum class DataType { HALF, FLOAT, DOUBLE, INT8, INT16, INT32, INT64 };

int GetElementSize(DataType type) {
  switch (type) {
  case DataType::HALF: return 16;
  case DataType::FLOAT: return 32;
  case DataType::DOUBLE: return 64;
  case DataType::INT8: return 8;
  case DataType::INT16: return 16;
  case DataType::INT32: return 32;
  case DataType::INT64: return 64;
  default: return 0;
  }
}

// cutensor始终有数据的所有权,且数据始终是连续的
class CuTensor {
 public:
  CuTensor();
  // 支持vector初始化
  explicit CuTensor(const std::vector<int> &shapes, DataType dataType = DataType::FLOAT);
  // 支持initializer_list初始化
  explicit CuTensor(const std::initializer_list<int> &shapes, DataType dataType = DataType::FLOAT);
  // 从已有GPU内存初始化cutensor
  CuTensor(void *ptr, const std::vector<int> shapes, DataType dataType = DataType::FLOAT);
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
  void *data();
  void *data() const;
  void fromcpu(const CTensorfi &t);
  // 将数据拷贝到CPU内存
  CTensorfi cpu();
  std::vector<int> shapes() const;
  std::vector<int> strides() const;
  uint64_t size() const;
  uint64_t byteSize() const;
  cudnnTensorDescriptor_t tensor_desc() const;

 private:
  void *data_;
  DataType data_type_;
  short element_size_;
  long size_;
  bool is_tensor_;
  bool hold_data_;
  int dim_num_;               // 实际维度数
  std::vector<int> shapes_;   // 维度
  std::vector<int> strides_;  // 每个维度步长
  TensorDescriptorRAII tensor_desc_raii_;
};

CuTensor::CuTensor() :
  data_(nullptr), data_type_(DataType::FLOAT), size_(0), element_size_(32), is_tensor_(false), hold_data_(false),
  dim_num_(0) {
  shapes_.clear();
  strides_.clear();
}

CuTensor::CuTensor(const std::vector<int> &shapes, DataType dataType) :
  shapes_(shapes), is_tensor_(true), hold_data_(true) {
  assert(shapes.size() > 0);
  data_type_ = dataType;
  element_size_ = GetElementSize(dataType);
  strides_.resize(shapes.size());
  size_ = shapes_[0];
  strides_[shapes.size() - 1] = 1;
  for (int i = shapes_.size() - 1; i > 0; i--) {
    // assert(shapes_[i]>0);
    size_ *= shapes_[i];
    strides_[i - 1] = strides_[i] * shapes_[i];
  }
  cudaMalloc((void **)&data_, element_size_ * size_);
}

// 支持initializer_list初始化
CuTensor::CuTensor(const std::initializer_list<int> &shapes, DataType dataType) :
  shapes_(shapes), is_tensor_(true), hold_data_(true) {
  assert(shapes.size() > 0);
  data_type_ = dataType;
  element_size_ = GetElementSize(dataType);
  strides_.resize(shapes.size());
  size_ = shapes_[0];
  strides_[shapes.size() - 1] = 1;
  for (int i = shapes_.size() - 1; i > 0; i--) {
    // assert(shapes_[i]>0);
    size_ *= shapes_[i];
    strides_[i - 1] = strides_[i] * shapes_[i];
  }
  cudaMalloc((void **)&data_, element_size_ * size_);
}

CuTensor::CuTensor(void *ptr, const std::vector<int> shapes, DataType dataType) :
  shapes_(shapes), is_tensor_(true), hold_data_(false) {
  data_type_ = dataType;
  element_size_ = GetElementSize(dataType);
  data_ = ptr;
  strides_.resize(shapes.size());
  size_ = shapes_[0];
  strides_[shapes.size() - 1] = 1;
  for (int i = shapes_.size() - 1; i > 0; i--) {
    // assert(shapes_[i]>0);
    size_ *= shapes_[i];
    strides_[i - 1] = strides_[i] * shapes_[i];
  }
}

// 拷贝构造函数
CuTensor::CuTensor(const CuTensor &t) {
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  data_type_ = t.data_type_;
  element_size_ = t.element_size_;
  is_tensor_ = t.is_tensor_;
  hold_data_ = t.hold_data_;
  cudaMalloc((void **)&data_, element_size_ * size_);
  cudaMemcpy(data_, t.data_, size_ * element_size_, cudaMemcpyDeviceToDevice);
}

// 移动构造函数
CuTensor::CuTensor(CuTensor &&t) noexcept {
  shapes_ = std::move(t.shapes_);
  strides_ = std::move(t.strides_);
  size_ = t.size_;
  data_type_ = t.data_type_;
  element_size_ = t.element_size_;
  data_ = t.data_;
  t.data_ = nullptr;
  is_tensor_ = t.is_tensor_;
  hold_data_ = t.hold_data_;
  t.hold_data_ = false;
  tensor_desc_raii_ = std::move(t.tensor_desc_raii_);
  t.is_tensor_ = false;
}

// 赋值构造函数
CuTensor &CuTensor::operator=(const CuTensor &t) {
  if (this == &t) return *this;
  // copy & swap
  CuTensor temp{t};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(data_type_, temp.data_type_);
  std::swap(element_size_, temp.element_size_);
  std::swap(data_, temp.data_);
  std::swap(tensor_desc_raii_, temp.tensor_desc_raii_);
  std::swap(is_tensor_, temp.is_tensor_);
  hold_data_ = t.hold_data_;
  return *this;
}

// 移动赋值构造函数
CuTensor &CuTensor::operator=(CuTensor &&t) noexcept {
  if (this == &t) return *this;
  // copy & swap
  CuTensor temp{std::move(t)};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(data_type_, temp.data_type_);
  std::swap(element_size_, temp.element_size_);
  std::swap(data_, temp.data_);
  std::swap(tensor_desc_raii_, temp.tensor_desc_raii_);
  std::swap(is_tensor_, temp.is_tensor_);
  std::swap(hold_data_, t.hold_data_);
  return *this;
}

CuTensor::~CuTensor() {
  if (hold_data_) CuMemDeleter<void>()(data_);
}

// 调整tensor大小，会重新分配内存，保证内存连续
void CuTensor::resize(const std::vector<int> shapes) {
  if (shapes == shapes_) return;
  CuTensor newTensor(shapes, data_type_);
  std::swap(size_, newTensor.size_);
  std::swap(shapes_, newTensor.shapes_);
  std::swap(strides_, newTensor.strides_);
  std::swap(data_type_, newTensor.data_type_);
  std::swap(element_size_, newTensor.element_size_);
  std::swap(data_, newTensor.data_);
  std::swap(tensor_desc_raii_, newTensor.tensor_desc_raii_);
  std::swap(is_tensor_, newTensor.is_tensor_);
  std::swap(hold_data_, newTensor.hold_data_);
}

void CuTensor::resize(const std::initializer_list<int> &shapes) {
  std::vector<int> s{shapes};
  resize(s);
}

// 清除数据
void CuTensor::clear() {
  shapes_.clear();
  strides_.clear();
  size_ = 0;
  if (hold_data_) CuMemDeleter<void>()(data_);
  is_tensor_ = false;
  hold_data_ = false;
}

void *CuTensor::data() {
  return data_;
}

void *CuTensor::data() const {
  return data_;
}

std::vector<int> CuTensor::shapes() const {
  return shapes_;
}

std::vector<int> CuTensor::strides() const {
  return strides_;
}

uint64_t CuTensor::size() const {
  return size_;
}

uint64_t CuTensor::byteSize() const {
  return size_ * element_size_;
}

void CuTensor::fromcpu(const CTensorfi &t) {
  if (shapes_ != t.shapes()) {
    printf("cpu/gpu shapes not matchs");
    return;
  }
  if (data_type_ != DataType::FLOAT) {
    printf("cpu/gpu dataType not matchs");
    return;
  }
  {
    shapes_ = t.shapes();
    strides_ = t.strides();
    size_ = t.size();
  }
  checkCudaErrors(cudaMemcpy(data_, t.data(), size_ * element_size_, cudaMemcpyHostToDevice));
}

CTensorfi CuTensor::cpu() {
  CTensorfi t(shapes_);
  checkCudaErrors(cudaMemcpy(t.data(), data_, size_ * element_size_, cudaMemcpyDeviceToHost));
  return t;
}

cudnnTensorDescriptor_t CuTensor::tensor_desc() const {
  if (shapes_.size() <= 4) {
    std::array<int, 4> nchw{1, 1, 1, 1};
    for (int i = 0; i < shapes_.size(); i++)
      nchw[i] = shapes_[i];
    cudnnSetTensor4dDescriptor(tensor_desc_raii_.tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, nchw[0], nchw[1],
                               nchw[2], nchw[3]);
  }
  return tensor_desc_raii_.tensor_desc_;
}

};  // namespace CUDA_NAMESPACE

#endif