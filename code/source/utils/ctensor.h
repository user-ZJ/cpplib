/*
 * @Author: zack 
 * @Date: 2021-10-05 10:26:48 
 * @Last Modified by: zack
 * @Last Modified time: 2021-10-21 16:51:30
 */
#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <cstring>
#include <utility>
#include <initializer_list>
#include <memory>

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
    shapes_ = shapes;
    strides_.resize(shapes.size());
    size_ = shapes_[0];
    strides_[shapes.size()-1] = 1;
    for (int i=shapes_.size()-1; i>0; i--) {
      assert(shapes_[i]>0);
      size_ *= shapes_[i];
      strides_[i-1] = strides_[i]*shapes_[i];
    }
    data_.reset(new T[size_]);
    memset(data_.get(),0,size_*sizeof(T));
  }
  // 支持initializer_list初始化
  explicit CTensor(const std::initializer_list<I> &shapes) {
    assert(shapes.size()>0);
    shapes_ = shapes;
    strides_.resize(shapes.size());
    size_ = shapes_[0];
    strides_[shapes.size()-1] = 1;
    for (int i=shapes_.size()-1; i>0; i--) {
      assert(shapes_[i]>0);
      size_ *= shapes_[i];
      strides_[i-1] = strides_[i]*shapes_[i];
    }
    data_.reset(new T[size_]);
    memset(data_.get(),0,size_*sizeof(T));
  }
  // 拷贝构造函数
  CTensor(const CTensor &t) {
    shapes_ = t.shapes_;
    strides_ = t.strides_;
    size_ = t.size_;
    data_.reset(new T[size_]);
    ::memcpy(data_.get(),t.data_.get(),size_*sizeof(T));
  }

  // 移动构造函数
  CTensor(CTensor &&t) {
    shapes_ = t.shapes_;
    strides_ = t.strides_;
    size_ = t.size_;
    data_ = t.data_;
    t.data_=nullptr;
  }

  // 赋值构造函数
  CTensor &operator=(const CTensor &t) {
    if(this==&t)
      return *this;
    // if(data_!=nullptr)
    //   delete [] data_;
    shapes_ = t.shapes_;
    strides_ = t.strides_;
    size_ = t.size_;
    data_.reset(new T[size_]);
    ::memcpy(data_.get(),t.data_.get(),size_*sizeof(T));
    return *this;
  }

  // 移动赋值构造函数
  CTensor &operator=(CTensor &&t) {
    if(this==&t)
      return *this;
    if(data_!=nullptr)
      data_.reset();
    shapes_ = t.shapes_;
    strides_ = t.strides_;
    size_ = t.size_;
    data_ = t.data_;
    t.data_ = nullptr;
    return *this;
  }

  ~CTensor() {
    // if (data_ != nullptr)
    //   delete [] (T *)data_;
    // data_ = nullptr;
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
  void resize(const std::initializer_list<I> &shapes) {
    CTensor newTensor(shapes);
    std::swap(size_, newTensor.size_);
    std::swap(shapes_, newTensor.shapes_);
    std::swap(strides_, newTensor.strides_);
    std::swap(data_, newTensor.data_);
  }
  // 清除数据
  void clear(){
    // if (data_ != nullptr){
    //   delete [] (T *)data_;
    //   data_ = nullptr;
    // }
    shapes_.clear();
    strides_.clear();
    size_ = 0;
  }

  // 访问数据
  T &at(const std::initializer_list<I> &indexs){
    assert(indexs.size()==shapes_.size());
    T *ptr = data_.get();
    int i = 0;
    for(auto d:indexs){
      ptr += d*strides_[i++];
    }
    return *ptr;
  }

  const T &at(const std::initializer_list<I> &indexs) const{
    assert(indexs.size()==shapes_.size());
    T *ptr = data_.get();
    int i = 0;
    for(auto d:indexs){
      assert(d<shapes_[i]); // 检查是否越界，防止踩内存
      ptr += d*strides_[i++];
    }
    return *ptr;
  }

  T *data() { return data_.get(); }
  T *data() const { return data_.get(); }

  std::vector<I> shapes() const { return shapes_; }

  std::vector<I> strides() const { return strides_; }

  uint64_t size() const { return size_; }
  uint64_t byteSize() const { return size_*sizeof(T); }

  std::vector<T> vector(){
    std::vector<T> out(size_);
    memcpy(out.data(),data_.get(),size_*sizeof(T));
    return out;
  }

#if DEBUG
  // dump data to file,only for debug
  void dump2File(const char *filename) {
    std::ofstream out(filename);
    if (out.is_open()) {
      //shapes
      out<<"[";
      for(auto s:shapes_)
        out<<s<<",";
      out<<"]\n";
      // data
      if (shapes_.size() == 1) {
        for (int64_t i = 0; i < shapes_[0];i++) {
          out << std::to_string(*(data_.get()+i)) << " ";
        }
      } else if (shapes_.size() == 2) {
        for (int64_t i=0;i<shapes_[0];i++) {
          for (int j=0;j<shapes_[1];j++)
            out << std::to_string(*(data_.get()+i*strides_[0]+j)) << " ";
          out << "\n";
        }
      }else{
        size_t row=shapes_[0],col=shapes_[shapes_.size()-1];
        for(int i=1;i<shapes_.size()-1;i++)
          row *= shapes_[i];
        for (int64_t i=0;i<row;i++) {
          for (int j=0;j<col;j++)
            out << std::to_string(*(data_.get()+i*col+j)) << " ";
          out << "\n";
        }
      }
      out << "\n";
      out.close();
    }
  }
#endif

private:
  std::shared_ptr<T> data_;
  std::vector<I> shapes_;
  std::vector<I> strides_;
  long size_;
};

typedef CTensor<float,int64_t> CTensorfl;
typedef CTensor<float,int32_t> CTensorfi;
typedef CTensor<int64_t,int64_t> CTensorll;
typedef CTensor<int32_t,int32_t> CTensorii;
typedef CTensor<int32_t,int64_t> CTensoril;

};
