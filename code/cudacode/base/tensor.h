/*
 * @Author: zack
 * @Date: 2021-10-05 10:26:48
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 15:21:49
 */
#ifndef BASE_CTENSOR_UTIL_H_
#define BASE_CTENSOR_UTIL_H_
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

// 用于传递给不同框架的数据结构
// ctensor始终有数据的所有权,且数据始终是连续的
class CTensor {
 public:
  CTensor();
  // 支持vector初始化
  explicit CTensor(const std::vector<int> &shapes,DataType data_type = DataType::FLOAT);
  // 支持initializer_list初始化
  explicit CTensor(const std::initializer_list<int> &shapes,DataType data_type = DataType::FLOAT);
  // 拷贝构造函数
  CTensor(const CTensor &t);
  // 移动构造函数
  CTensor(CTensor &&t) noexcept;
  // 赋值构造函数
  CTensor &operator=(const CTensor &t);
  // 移动赋值构造函数
  CTensor &operator=(CTensor &&t) noexcept;

  ~CTensor();
  // 调整tensor大小，会重新分配内存，保证内存连续
  void resize(const std::vector<int> shapes);
  void resize(const std::initializer_list<int> &shapes);
  // 清除数据
  void clear();
  // 访问数据
  template <typename T>
  T &at(const std::initializer_list<int> &indexs,T *p=nullptr);
  template <typename T>
  const T &at(const std::initializer_list<int> &indexs,T *p=nullptr) const;
  void *data();
  void *data() const;
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
  void *data_;
  DataType data_type_;
  std::vector<int> shapes_;
  std::vector<int> strides_;
  long size_;
  long byte_size_;
};


CTensor::CTensor() {
  data_ = nullptr;
  data_type_ = DataType::FLOAT;
  size_ = 0;
  shapes_.clear();
  strides_.clear();
}


CTensor::CTensor(const std::vector<int> &shapes,DataType data_type) {
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
  data_type_ = data_type;
  byte_size_ = size_*GetElementSize(data_type_);
  data_ = malloc(byte_size_);
  memset(data_, 0, byte_size_);
}

// 支持initializer_list初始化

CTensor::CTensor(const std::initializer_list<int> &shapes,DataType data_type) {
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
  data_type_ = data_type;
  byte_size_ = size_*GetElementSize(data_type_);
  data_ = malloc(byte_size_);
  memset(data_, 0, byte_size_);
}

// 拷贝构造函数

CTensor::CTensor(const CTensor &t) {
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  byte_size_ = t.byte_size_;
  data_type_ = t.data_type_;
  data_ = malloc(byte_size_);
  ::memcpy(data_, t.data_, byte_size_);
}

// 移动构造函数

CTensor::CTensor(CTensor &&t) noexcept {
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  byte_size_ = t.byte_size_;
  data_type_ = t.data_type_;
  data_ = t.data_;
  t.data_ = nullptr;
}

// 赋值构造函数

CTensor &CTensor::operator=(const CTensor &t) {
  if (this == &t) return *this;
  // copy & swap
  CTensor temp{t};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(byte_size_, temp.byte_size_);
  std::swap(data_type_, temp.data_type_);
  std::swap(data_, temp.data_);
  return *this;
}

// 移动赋值构造函数

CTensor &CTensor::operator=(CTensor &&t) noexcept {
  if (this == &t) return *this;
  // copy & swap
  CTensor temp{std::move(t)};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(byte_size_, temp.byte_size_);
  std::swap(data_type_, temp.data_type_);
  std::swap(data_, temp.data_);
  return *this;
}


CTensor::~CTensor() {
  if(data_) free(data_);
}
// 调整tensor大小，会重新分配内存，保证内存连续

void CTensor::resize(const std::vector<int> shapes) {
  if (shapes == shapes_) return;
  CTensor newTensor(shapes);
  std::swap(size_, newTensor.size_);
  std::swap(byte_size_, newTensor.byte_size_);
  std::swap(data_type_, newTensor.data_type_);
  std::swap(shapes_, newTensor.shapes_);
  std::swap(strides_, newTensor.strides_);
  std::swap(data_, newTensor.data_);
}


void CTensor::resize(const std::initializer_list<int> &shapes) {
  CTensor newTensor(shapes);
  std::swap(size_, newTensor.size_);
  std::swap(byte_size_, newTensor.byte_size_);
  std::swap(data_type_, newTensor.data_type_);
  std::swap(shapes_, newTensor.shapes_);
  std::swap(strides_, newTensor.strides_);
  std::swap(data_, newTensor.data_);
}

// 清除数据

void CTensor::clear() {
  shapes_.clear();
  strides_.clear();
  size_ = 0;
  byte_size_=0;
  free(data_);
  data_=nullptr;
}

// 访问数据
template <typename T>
T &CTensor::at(const std::initializer_list<int> &indexs,T *p) {
  assert(indexs.size() == shapes_.size());
  char *ptr = static_cast(char*)data_;
  int i = 0;
  for (auto d : indexs) {
    assert(d < shapes_[i]);  // 检查是否越界，防止踩内存
    ptr += d * strides_[i++]*GetElementSize(data_type_);
  }
  return *((T *)ptr);
}

template <typename T>
const T &CTensor::at(const std::initializer_list<int> &indexs,T *p) const {
  assert(indexs.size() == shapes_.size());
  char *ptr = static_cast(char*)data_;
  int i = 0;
  for (auto d : indexs) {
    ptr += d * strides_[i++]*GetElementSize(data_type_);
  }
  return *ptr;
}


void *CTensor::data() {
  return data_;
}


void *CTensor::data() const {
  return data_;
}


std::vector<int> CTensor::shapes() const {
  return shapes_;
}


std::vector<int> CTensor::strides() const {
  return strides_;
}


uint64_t CTensor::size() const {
  return size_;
}

uint64_t CTensor::byteSize() const {
  return byte_size_;
}

template <typename T>
void CTensor::dump2File(const char *filename) const {
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
        out << std::to_string(*((T *)data_ + i)) << " ";
      }
    } else if (shapes_.size() == 2) {
      for (int64_t i = 0; i < shapes_[0]; i++) {
        for (int j = 0; j < shapes_[1]; j++)
          out << std::to_string(*((T *)data_ + i * strides_[0] + j)) << " ";
        out << "\n";
      }
    } else {
      size_t row = shapes_[0], col = shapes_[shapes_.size() - 1];
      for (int i = 1; i < shapes_.size() - 1; i++)
        row *= shapes_[i];
      for (int64_t i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
          out << std::to_string(*((T *)data_ + i * col + j)) << " ";
        out << "\n";
      }
    }
    out << "\n";
    out.close();
  }
}


int CTensor::writeFile(const char *filename) const{
  std::ofstream out(filename, std::ios::out|std::ios::binary);
  out.write((char*)data_,byte_size_);
  out.close();
  return 0;
}


int CTensor::readFile(const char *filename){
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  in.read((char*)data_, byte_size_);
  in.close();
  return 0;
}

};  // namespace BASE_NAMESPACE

#endif