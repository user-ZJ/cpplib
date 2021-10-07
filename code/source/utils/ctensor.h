/*
 * @Author: zack 
 * @Date: 2021-10-05 10:26:48 
 * @Last Modified by: zack
 * @Last Modified time: 2021-10-05 11:07:46
 */
#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <cstring>

namespace BASE_NAMESPACE {

// 用于传递给不同框架的数据结构
// ctensor始终有数据的所有权,且数据始终是连续的
template <typename T,typename I> 
class CTensor {
public:
  CTensor() {
    data_ = nullptr;
    size_ = 0;
    shapes_.clear();
    strides_.clear();
  }
  // 支持vector初始化
  explicit CTensor(const std::vector<I> &shapes) {
    assert(shapes.size()>0);
    shapes_.resize(shapes.size());
    strides_.resize(shapes.size());
    for (int i = 0; i < shapes.size(); i++) {
      assert(shapes[i]>0);
      shapes_[i] = shapes[i];
      if (i == 0) {
        size_ = shapes[i];
        strides_[shapes.size() - i - 1] = 1;
      } else {
        strides_[shapes.size() - i - 1] = size_;
        size_ *= shapes[i];
      }
    }
    data_ = new T[size_];
  }
  // 支持initializer_list初始化
  explicit CTensor(const std::initializer_list<I> &shapes) {
    assert(shapes.size()>0);
    shapes_.reserve(shapes.size());
    strides_.reserve(shapes.size());
    size_ = 1;
    for (auto s:shapes) {
      assert(s>0);
      shapes_.push_back(s);
      strides_.push_back(size_);
      size_ *= s;
    }
    std::reverse(strides_.begin(),strides_.end());
    data_ = new T[size_];
  }
  // 拷贝构造函数
  CTensor(const CTensor &t) {
    shapes_ = t.shapes_;
    strides_ = t.strides_;
    size_ = t.size_;
    data_ = new T[size_];
    ::memcpy(data_,t.data_,size_*sizeof(T));
  }
  // 复制构造函数
  CTensor &operator=(const CTensor &t) {
    if(this==&t)
      return *this;
    if(data_!=nullptr)
      delete data_;
    shapes_ = t.shapes_;
    strides_ = t.strides_;
    size_ = t.size_;
    data_ = new T[size_];
    ::memcpy(data_,t.data_,size_*sizeof(T));
    return *this;
  }

  ~CTensor() {
    if (data_ != nullptr)
      delete (T *)data_;
  }
  // 调整tensor大小，会重新分配内存，保证内存连续
  void resize(const std::vector<I> shapes) {
    if(shapes == shapes_)
      return;
    CTensor newTensor(shapes);
    std::swap(size_, newTensor.size_);
    std::swap(shapes_, newTensor.shapes_);
    std::swap(strides_, newTensor.strides_);
    std::swap(data_, newTensor.data_);
  }

  T *data() { return data_; }
  T *data() const { return data_; }

  std::vector<I> shapes() const { return shapes_; }

  std::vector<I> strides() const { return strides_; }

  long size() const { return size_; }

private:
  T *data_;
  std::vector<I> shapes_;
  std::vector<I> strides_;
  long size_;
};

typedef CTensor<float,int64_t> CTensor64;
typedef CTensor<float,int32_t> CTensor32;

}; // namespace DMAI
