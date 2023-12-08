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
#include <cmath>

namespace BASE_NAMESPACE {


// 用于传递给不同框架的数据结构
// ctensor始终有数据的所有权,且数据始终是连续的
template <typename T, typename I>
class CTensor {
 public:
  CTensor();
  // 支持vector初始化
  explicit CTensor(const std::vector<I> &shapes);
  // 支持initializer_list初始化
  explicit CTensor(const std::initializer_list<I> &shapes);
  // 拷贝构造函数
  CTensor(const CTensor &t);
  // 移动构造函数
  CTensor(CTensor &&t) noexcept;
  // 赋值构造函数
  CTensor &operator=(const CTensor &t);
  // 移动赋值构造函数
  CTensor &operator=(CTensor &&t) noexcept;

  template <typename TS, typename IS>
  CTensor &copyFrom(const CTensor<TS, IS> &t);

  ~CTensor();
  // 调整tensor大小，会重新分配内存，保证内存连续
  void resize(const std::vector<I> shapes);
  void resize(const std::initializer_list<I> &shapes);
  // 清除数据
  void clear();
  // 访问数据
  T &at(const std::initializer_list<I> &indexs);
  const T &at(const std::initializer_list<I> &indexs) const;
  T *data();
  T *data() const;
  std::vector<I> shapes() const;
  std::vector<I> strides() const;
  uint64_t size() const;
  uint64_t byteSize() const;
  std::vector<T> vector();
  // dump data to file,only for debug
  void dump2File(const char *filename) const;
  int writeFile(const char *filename) const;
  int readFile(const char *filename);

 private:
  std::shared_ptr<T> data_;
  std::vector<I> shapes_;
  std::vector<I> strides_;
  long size_;
};

template <typename T, typename I>
CTensor<T, I>::CTensor() {
  data_ = nullptr;
  size_ = 0;
  shapes_.clear();
  strides_.clear();
}

template <typename T, typename I>
CTensor<T, I>::CTensor(const std::vector<I> &shapes) {
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
  data_.reset(new T[size_]);
  memset(data_.get(), 0, size_ * sizeof(T));
}

// 支持initializer_list初始化
template <typename T, typename I>
CTensor<T, I>::CTensor(const std::initializer_list<I> &shapes) {
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
  data_.reset(new T[size_]);
  memset(data_.get(), 0, size_ * sizeof(T));
}

// 拷贝构造函数
template <typename T, typename I>
CTensor<T, I>::CTensor(const CTensor &t) {
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  data_.reset(new T[size_]);
  ::memcpy(data_.get(), t.data_.get(), size_ * sizeof(T));
}

// 移动构造函数
template <typename T, typename I>
CTensor<T, I>::CTensor(CTensor &&t) noexcept {
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  data_ = t.data_;
  t.data_ = nullptr;
}

// 赋值构造函数
template <typename T, typename I>
CTensor<T, I> &CTensor<T, I>::operator=(const CTensor &t) {
  if (this == &t) return *this;
  // copy & swap
  CTensor temp{t};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(data_, temp.data_);
  return *this;
}

// 移动赋值构造函数
template <typename T, typename I>
CTensor<T, I> &CTensor<T, I>::operator=(CTensor &&t) noexcept {
  if (this == &t) return *this;
  // copy & swap
  CTensor temp{std::move(t)};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(data_, temp.data_);
  return *this;
}

template <typename T, typename I>
template <typename TS, typename IS>
CTensor<T, I> &CTensor<T, I>::copyFrom(const CTensor<TS, IS> &t) {
  shapes_.resize(t.shapes().size());
  for (int i = 0; i < t.shapes().size(); i++)
    shapes_[i] = static_cast<I>(t.shapes()[i]);
  strides_.resize(t.strides().size());
  for (int i = 0; i < t.strides().size(); i++)
    strides_[i] = static_cast<I>(t.strides()[i]);
  size_ = t.size();
  data_.reset(new T[size_]);
  for (int i = 0; i < size(); i++)
    *(data_.get() + i) = static_cast<T>(*(t.data() + i));
  return *this;
}

template <typename T, typename I>
CTensor<T, I>::~CTensor() {}
// 调整tensor大小，会重新分配内存，保证内存连续
template <typename T, typename I>
void CTensor<T, I>::resize(const std::vector<I> shapes) {
  if (shapes == shapes_) return;
  CTensor newTensor(shapes);
  std::swap(size_, newTensor.size_);
  std::swap(shapes_, newTensor.shapes_);
  std::swap(strides_, newTensor.strides_);
  std::swap(data_, newTensor.data_);
}

template <typename T, typename I>
void CTensor<T, I>::resize(const std::initializer_list<I> &shapes) {
  CTensor newTensor(shapes);
  std::swap(size_, newTensor.size_);
  std::swap(shapes_, newTensor.shapes_);
  std::swap(strides_, newTensor.strides_);
  std::swap(data_, newTensor.data_);
}

// 清除数据
template <typename T, typename I>
void CTensor<T, I>::clear() {
  shapes_.clear();
  strides_.clear();
  size_ = 0;
}

// 访问数据
template <typename T, typename I>
T &CTensor<T, I>::at(const std::initializer_list<I> &indexs) {
  assert(indexs.size() == shapes_.size());
  T *ptr = data_.get();
  int i = 0;
  for (auto d : indexs) {
    ptr += d * strides_[i++];
  }
  return *ptr;
}

template <typename T, typename I>
const T &CTensor<T, I>::at(const std::initializer_list<I> &indexs) const {
  assert(indexs.size() == shapes_.size());
  T *ptr = data_.get();
  int i = 0;
  for (auto d : indexs) {
    assert(d < shapes_[i]);  // 检查是否越界，防止踩内存
    ptr += d * strides_[i++];
  }
  return *ptr;
}

template <typename T, typename I>
T *CTensor<T, I>::data() {
  return data_.get();
}

template <typename T, typename I>
T *CTensor<T, I>::data() const {
  return data_.get();
}

template <typename T, typename I>
std::vector<I> CTensor<T, I>::shapes() const {
  return shapes_;
}

template <typename T, typename I>
std::vector<I> CTensor<T, I>::strides() const {
  return strides_;
}

template <typename T, typename I>
uint64_t CTensor<T, I>::size() const {
  return size_;
}
template <typename T, typename I>
uint64_t CTensor<T, I>::byteSize() const {
  return size_ * sizeof(T);
}

template <typename T, typename I>
std::vector<T> CTensor<T, I>::vector() {
  std::vector<T> out(size_);
  memcpy(out.data(), data_.get(), size_ * sizeof(T));
  return out;
}
template <typename T, typename I>
void CTensor<T, I>::dump2File(const char *filename) const {
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
        out << std::to_string(*(data_.get() + i)) << " ";
      }
    } else if (shapes_.size() == 2) {
      for (int64_t i = 0; i < shapes_[0]; i++) {
        for (int j = 0; j < shapes_[1]; j++)
          out << std::to_string(*(data_.get() + i * strides_[0] + j)) << " ";
        out << "\n";
      }
    } else {
      size_t row = shapes_[0], col = shapes_[shapes_.size() - 1];
      for (int i = 1; i < shapes_.size() - 1; i++)
        row *= shapes_[i];
      for (int64_t i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
          out << std::to_string(*(data_.get() + i * col + j)) << " ";
        out << "\n";
      }
    }
    out << "\n";
    out.close();
  }
}

template <typename T, typename I>
int CTensor<T, I>::writeFile(const char *filename) const{
  std::ofstream out(filename, std::ios::out|std::ios::binary);
  out.write((char*)data_.get(),size_*sizeof(T));
  out.close();
  return 0;
}

template <typename T, typename I>
int CTensor<T, I>::readFile(const char *filename){
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  in.read((char*)data_.get(), size_*sizeof(T));
  in.close();
  return 0;
}

typedef CTensor<float, int64_t> CTensorfl;
typedef CTensor<float, int32_t> CTensorfi;
typedef CTensor<int64_t, int64_t> CTensorll;
typedef CTensor<int32_t, int32_t> CTensorii;
typedef CTensor<int32_t, int64_t> CTensoril;


inline int adaptive_avg_pool1d(const CTensorfl &input,const int &output_size,CTensorfl *out){
  // input NxLxC output Nxoutput_sizexC
  auto input_shape = input.shapes();
  int input_size = input_shape[1];
  out->resize({input_shape[0],output_size,input_shape[2]});
  auto start_index = [&input_size,&output_size](int curr_i){ return (int)(std::floor((curr_i *1.0* input_size) / output_size));};
  auto end_index = [&input_size,&output_size](int curr_i){ return (int)(std::ceil(((curr_i+1) *1.0* input_size) / output_size));};
  for(int j=0;j<input_shape[2];j++){
    float sum=0;
    int start=0,end=0;
    for(int i=0;i<output_size;i++){
      int window_start = start_index(i),window_end=end_index(i);
      while(start<window_start) sum-=input.at({0,start++,j});
      while(end<window_end) sum+=input.at({0,end++,j});
      out->at({0,i,j}) = sum/(window_end-window_start);
    }
  }
  return 0;
}

inline int length_regulate(const CTensorfl &memories, const CTensorll &durs, CTensorfl *out) {
  // memories: [1, T, *]
  // durs: [1, T]
  // CHECK_EQ(durs.shapes()[0], 1);
  // CHECK_EQ(memories.shapes()[0], 1);
  // CHECK_EQ(memories.shapes()[1], durs.shapes()[1]);
  int64_t sum = 0;
  for (int64_t i = 0; i < durs.shapes()[1]; i++) {
    sum += durs.at({0, i});
  }
  out->resize({1, sum, memories.shapes()[2]});
  int64_t repeat_size = memories.shapes()[2] * sizeof(float);
  auto strides = out->strides();
  int64_t s_index = 0, t_index = 0;
  for (int64_t i = 0; i < durs.shapes()[1]; i++) {
    s_index = i;
    for (int64_t j = 0; j < durs.at({0, i}); j++) {  // repeat data
      memcpy(out->data() + t_index * strides[1], memories.data() + s_index * strides[1], repeat_size);
      t_index++;
    }
  }
  return 0;
}

};  // namespace BASE_NAMESPACE

#endif