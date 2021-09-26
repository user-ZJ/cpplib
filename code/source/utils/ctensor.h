#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <cstring>

namespace DMAI {

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
  explicit CTensor(std::vector<I> shapes) {
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


  CTensor(const CTensor &t) {
    shapes_ = t.shapes_;
    strides_ = t.strides_;
    size_ = t.size_;
    data_ = new T[size_];
    ::memcpy(data_,t.data_,size_*sizeof(T));
  }

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
  
  int resize(const std::vector<I> shapes) {
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

}; // namespace DMAI
