/*
 * @Author: zack
 * @Date: 2021-10-05 10:26:48
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 15:21:49
 */
#ifndef BASE_NDTensor_UTIL_H_
#define BASE_NDTensor_UTIL_H_
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include "cuhelper.h"
using namespace CUDA_NAMESPACE;
#endif

namespace CUDA_NAMESPACE {

template <typename T>
struct FreeDeleter {
  void operator()(T *ptr) const {
    free(ptr);
  }
};

enum class DataType { HALF, FLOAT, DOUBLE, INT8, INT16, INT32, INT64 };
enum class DeviceType { CPU, CUDA };

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

// 用于传递给不同框架的数据结构
// NDTensor始终有数据的所有权,且数据始终是连续的
class NDTensor {
 public:
  NDTensor();
  // 支持vector初始化
  explicit NDTensor(const std::vector<int> &shapes, DataType data_type = DataType::FLOAT,
                    DeviceType device_type = DeviceType::CPU);
  // 支持initializer_list初始化
  explicit NDTensor(const std::initializer_list<int> &shapes, DataType data_type = DataType::FLOAT,
                    DeviceType device_type = DeviceType::CPU);
  // 拷贝构造函数
  NDTensor(const NDTensor &t);
  // 移动构造函数
  NDTensor(NDTensor &&t) noexcept;
  // 赋值构造函数
  NDTensor &operator=(const NDTensor &t);
  // 移动赋值构造函数
  NDTensor &operator=(NDTensor &&t) noexcept;

  ~NDTensor();
  // 调整tensor大小，会重新分配内存，保证内存连续
  void resize(const std::vector<int> shapes);
  void resize(const std::initializer_list<int> &shapes);
  // 清除数据
  void clear();
#ifdef USE_CUDA
  void cpu();
  void cuda();
#endif
  // 访问数据
  template <typename T>
  T &at(const std::initializer_list<int> &indexs, T *p = nullptr);
  template <typename T>
  const T &at(const std::initializer_list<int> &indexs, T *p = nullptr) const;
  template <typename T>
  T *data();
  template <typename T>
  T *data() const;
  std::vector<int> shapes() const;
  std::vector<int> strides() const;
  uint64_t size() const;
  uint64_t byteSize() const;
  // dump data to file,only for debug
  template <typename T>
  void dump2File(const char *filename) const;
  int writeFile(const char *filename) const;
  int readFile(const char *filename);

 private:
  void *host_data_;
  void *device_data_;
#ifdef USE_CUDA
  UniqPtr<char> uniq_device_data_;
#endif
  DataType data_type_;
  DeviceType device_type_;
  std::vector<int> shapes_;
  std::vector<int> strides_;
  long size_;
  long byte_size_;
};

NDTensor::NDTensor() :
  host_data_(nullptr), device_data_(nullptr), data_type_(DataType::FLOAT), device_type_(DeviceType::CPU), size_(0) {}

NDTensor::NDTensor(const std::vector<int> &shapes, DataType data_type, DeviceType device_type) :
  data_type_(data_type), device_type_(device_type) {
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
  byte_size_ = size_ * GetElementSize(data_type);
  if (device_type == DeviceType::CPU) {
    host_data_ = malloc(byte_size_);
    memset(host_data_, 0, byte_size_);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    uniq_device_data_ = mallocCudaMem<char>(byte_size_);
    device_data_ = (void *)uniq_device_data_.get();
  }
#endif
}

// 支持initializer_list初始化

NDTensor::NDTensor(const std::initializer_list<int> &shapes, DataType data_type, DeviceType device_type) :
  data_type_(data_type), device_type_(device_type) {
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
  byte_size_ = size_ * GetElementSize(data_type);
  if (device_type == DeviceType::CPU) {
    host_data_ = malloc(byte_size_);
    memset(host_data_, 0, byte_size_);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    uniq_device_data_ = mallocCudaMem<char>(byte_size_);
    device_data_ = (void *)uniq_device_data_.get();
  }
#endif
}

// 拷贝构造函数

NDTensor::NDTensor(const NDTensor &t) {
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  byte_size_ = t.byte_size_;
  data_type_ = t.data_type_;
  device_type_ = t.device_type_;
  if (device_type_ == DeviceType::CPU) {
    host_data_ = malloc(byte_size_);
    memcpy(host_data_, t.host_data_, byte_size_);
  }
#ifdef UES_CUDA
  else if (device_type_ == DeviceType::CUDA) {
    uniq_device_data_ = mallocCudaMem<char>(byte_size_);
    device_data_ = (void *)uniq_device_data_.get();
    cudaMemcpy(device_data_, t.device_data_, byte_size_, cudaMemcpyDeviceToDevice);
  }
#endif
}

// 移动构造函数

NDTensor::NDTensor(NDTensor &&t) noexcept {
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  byte_size_ = t.byte_size_;
  data_type_ = t.data_type_;
  device_type_ = t.device_type_;
  if (device_type_ == DeviceType::CPU) {
    host_data_ = t.host_data_;
    t.host_data_ = nullptr;
  }
#ifdef UES_CUDA
  else if (device_type_ == DeviceType::CUDA) {
    uniq_device_data_ = std::move(t.uniq_device_data_);
    device_data_ = (void *)uniq_device_data_.get();
    t.device_data_ = nullptr;
  }
#endif
}

// 赋值构造函数

NDTensor &NDTensor::operator=(const NDTensor &t) {
  if (this == &t) return *this;
  // copy & swap
  NDTensor temp{t};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(byte_size_, temp.byte_size_);
  std::swap(data_type_, temp.data_type_);
  std::swap(host_data_, temp.host_data_);
#ifdef USE_CUDA
  uniq_device_data_.swap(temp.uniq_device_data_);
  std::swap(device_data_, temp.device_data_);
#endif
  return *this;
}

// 移动赋值构造函数

NDTensor &NDTensor::operator=(NDTensor &&t) noexcept {
  if (this == &t) return *this;
  // copy & swap
  NDTensor temp{std::move(t)};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(byte_size_, temp.byte_size_);
  std::swap(data_type_, temp.data_type_);
  std::swap(host_data_, temp.host_data_);
#ifdef USE_CUDA
  uniq_device_data_.swap(temp.uniq_device_data_);
  std::swap(device_data_, temp.device_data_);
#endif
  return *this;
}

NDTensor::~NDTensor() {
  if (host_data_) free(host_data_);
}
// 调整tensor大小，会重新分配内存，保证内存连续

void NDTensor::resize(const std::vector<int> shapes) {
  if (shapes == shapes_) return;
  NDTensor newTensor(shapes);
  std::swap(size_, newTensor.size_);
  std::swap(byte_size_, newTensor.byte_size_);
  std::swap(data_type_, newTensor.data_type_);
  std::swap(shapes_, newTensor.shapes_);
  std::swap(strides_, newTensor.strides_);
  std::swap(host_data_, newTensor.host_data_);
#ifdef USE_CUDA
  uniq_device_data_.swap(newTensor.uniq_device_data_);
  std::swap(device_data_, newTensor.device_data_);
#endif
}

void NDTensor::resize(const std::initializer_list<int> &shapes) {
  NDTensor newTensor(shapes);
  std::swap(size_, newTensor.size_);
  std::swap(byte_size_, newTensor.byte_size_);
  std::swap(data_type_, newTensor.data_type_);
  std::swap(shapes_, newTensor.shapes_);
  std::swap(strides_, newTensor.strides_);
  std::swap(host_data_, newTensor.host_data_);
#ifdef USE_CUDA
  uniq_device_data_.swap(newTensor.uniq_device_data_);
  std::swap(device_data_, newTensor.device_data_);
#endif
}

// 清除数据

void NDTensor::clear() {
  shapes_.clear();
  strides_.clear();
  size_ = 0;
  byte_size_ = 0;
  free(host_data_);
  host_data_ = nullptr;
#ifdef USE_CUDA
  uniq_device_data_.reset();
  device_data_ = nullptr;
#endif
}

#ifdef USE_CUDA
void NDTensor::cpu() {
  if (host_data_ == nullptr) host_data_ = malloc(byte_size_);
  cudaMemcpy(host_data_, device_data_, byte_size_, cudaMemcpyDeviceToHost);
  uniq_device_data_.reset();
  device_data_ = nullptr;
  device_type_ = DeviceType::CPU;
}
void NDTensor::cuda() {
  if (uniq_device_data_ == nullptr) {
    uniq_device_data_ = mallocCudaMem<char>(byte_size_);
    device_data_ = (void *)uniq_device_data_.get();
  }
  cudaMemcpy(device_data_, host_data_, byte_size_, cudaMemcpyHostToDevice);
  free(host_data_);
  host_data_ = nullptr;
  device_type_ = DeviceType::CUDA;
}
#endif

// 访问数据
template <typename T>
T &NDTensor::at(const std::initializer_list<int> &indexs, T *p) {
  assert(indexs.size() == shapes_.size());
  assert(device_type_==DeviceType::CPU);
  assert(sizeof(T)==GetElementSize(data_type_) && "data type mismatch");
  char *ptr = static_cast<char *>(host_data_);
  int i = 0;
  for (auto d : indexs) {
    assert(d < shapes_[i]);  // 检查是否越界，防止踩内存
    ptr += d * strides_[i++] * GetElementSize(data_type_);
  }
  return *((T *)ptr);
}

template <typename T>
const T &NDTensor::at(const std::initializer_list<int> &indexs, T *p) const {
  assert(indexs.size() == shapes_.size());
  assert(device_type_==DeviceType::CPU);
  char *ptr = static_cast<char *>(host_data_);
  int i = 0;
  for (auto d : indexs) {
    ptr += d * strides_[i++] * GetElementSize(data_type_);
  }
  return *ptr;
}

template <typename T>
T *NDTensor::data() {
  return static_cast<T *>(host_data_);
}

template <typename T>
T *NDTensor::data() const {
  return static_cast<T *>(host_data_);
}

std::vector<int> NDTensor::shapes() const {
  return shapes_;
}

std::vector<int> NDTensor::strides() const {
  return strides_;
}

uint64_t NDTensor::size() const {
  return size_;
}

uint64_t NDTensor::byteSize() const {
  return byte_size_;
}

template <typename T>
void NDTensor::dump2File(const char *filename) const {
  assert(device_type_==DeviceType::CPU);
  std::ofstream out(filename);
  if (out.is_open()) {
    // shapes
    out << "[";
    for (auto s : shapes_)
      out << s << ",";
    out << "]\n";
    // data
    if (shapes_.size() == 1) {
      for (int64_t i = 0; i < shapes_[0]; i++) {
        out << std::to_string(*((T *)host_data_ + i)) << " ";
      }
    } else if (shapes_.size() == 2) {
      for (int64_t i = 0; i < shapes_[0]; i++) {
        for (int j = 0; j < shapes_[1]; j++)
          out << std::to_string(*((T *)host_data_ + i * strides_[0] + j)) << " ";
        out << "\n";
      }
    } else {
      size_t row = shapes_[0], col = shapes_[shapes_.size() - 1];
      for (int i = 1; i < shapes_.size() - 1; i++)
        row *= shapes_[i];
      for (int64_t i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
          out << std::to_string(*((T *)host_data_ + i * col + j)) << " ";
        out << "\n";
      }
    }
    out << "\n";
    out.close();
  }
}

int NDTensor::writeFile(const char *filename) const {
  assert(device_type_==DeviceType::CPU);
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  out.write((char *)host_data_, byte_size_);
  out.close();
  return 0;
}

int NDTensor::readFile(const char *filename) {
  assert(device_type_==DeviceType::CPU);
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  in.read((char *)host_data_, byte_size_);
  in.close();
  return 0;
}

};  // namespace BASE_NAMESPACE

#endif