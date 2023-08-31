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

inline int GetElementSize(DataType type) {
  switch (type) {
  case DataType::HALF: return 2;
  case DataType::FLOAT: return 4;
  case DataType::DOUBLE: return 8;
  case DataType::INT8: return 1;
  case DataType::INT16: return 2;
  case DataType::INT32: return 4;
  case DataType::INT64: return 8;
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
  NDTensor cpu();
  NDTensor cuda();
#endif
  // 访问数据
  template <typename T>
  T &at(const std::initializer_list<int> &indexs);
  template <typename T>
  const T &at(const std::initializer_list<int> &indexs) const;
  template <typename T>
  T *data();
  template <typename T>
  T *data() const;
  std::vector<int> shapes() const;
  std::vector<int> strides() const;
  DataType getDataType() const { return data_type_; }
  DeviceType getDeviceType() const { return device_type_;}
  long size() const;
  long byteSize() const;
  // dump data to file,only for debug
  template <typename T>
  T dump2File(const char *filename) const;
  int writeFile(const char *filename) const;
  int readFile(const char *filename);
  int readBuff(const std::vector<char> buff);

 private:
  void *host_data_;
  std::unique_ptr<void,FreeDeleter<void>> uniq_host_data_;
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



};  // namespace CUDA_NAMESPACE

#endif