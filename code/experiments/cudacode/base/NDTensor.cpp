#include "NDTensor.h"
#include <iostream>

namespace CUDA_NAMESPACE {

NDTensor::NDTensor() :
  host_data_(nullptr), device_data_(nullptr), data_type_(DataType::FLOAT), device_type_(DeviceType::CPU), size_(0) {
    // std::cout<<"default construct"<<std::endl;
}

NDTensor::NDTensor(const std::vector<int> &shapes, DataType data_type, DeviceType device_type) :
  data_type_(data_type), device_type_(device_type) {
  assert(shapes.size() > 0);
//   std::cout<<"vector construct"<<std::endl;
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
    uniq_host_data_.reset(host_data_);
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
//   std::cout<<"ini list construct"<<std::endl;
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
    uniq_host_data_.reset(host_data_);
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
//   std::cout<<"copy construct"<<std::endl;
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  byte_size_ = t.byte_size_;
  data_type_ = t.data_type_;
  device_type_ = t.device_type_;
  if (device_type_ == DeviceType::CPU) {
    host_data_ = malloc(byte_size_);
    memcpy(host_data_, t.host_data_, byte_size_);
    uniq_host_data_.reset(host_data_);
  }
#ifdef USE_CUDA
  else if (device_type_ == DeviceType::CUDA) {
    uniq_device_data_ = mallocCudaMem<char>(byte_size_);
    device_data_ = (void *)uniq_device_data_.get();
    cudaMemcpy(device_data_, t.device_data_, byte_size_, cudaMemcpyDeviceToDevice);
  }
#endif
}

// 移动构造函数

NDTensor::NDTensor(NDTensor &&t) noexcept {
//   std::cout<<"move construct"<<std::endl;
  shapes_ = t.shapes_;
  strides_ = t.strides_;
  size_ = t.size_;
  byte_size_ = t.byte_size_;
  data_type_ = t.data_type_;
  device_type_ = t.device_type_;
  if (device_type_ == DeviceType::CPU) {
    uniq_host_data_ = std::move(t.uniq_host_data_);
    host_data_ = uniq_host_data_.get();
  }
#ifdef USE_CUDA
  else if (device_type_ == DeviceType::CUDA) {
    uniq_device_data_ = std::move(t.uniq_device_data_);
    device_data_ = (void *)uniq_device_data_.get();
    t.device_data_ = nullptr;
  }
#endif
}

// 赋值构造函数

NDTensor &NDTensor::operator=(const NDTensor &t) {
//   std::cout<<"copy sign"<<std::endl;
  if (this == &t) return *this;
  // copy & swap
  NDTensor temp{t};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(byte_size_, temp.byte_size_);
  std::swap(data_type_, temp.data_type_);
  std::swap(device_type_,temp.device_type_);
  uniq_host_data_.swap(temp.uniq_host_data_);
  std::swap(host_data_, temp.host_data_);
#ifdef USE_CUDA
  uniq_device_data_.swap(temp.uniq_device_data_);
  std::swap(device_data_, temp.device_data_);
#endif
  return *this;
}

// 移动赋值构造函数

NDTensor &NDTensor::operator=(NDTensor &&t) noexcept {
//   std::cout<<"move sign"<<std::endl;
  if (this == &t) return *this;
  // copy & swap
  NDTensor temp{std::move(t)};
  std::swap(shapes_, temp.shapes_);
  std::swap(strides_, temp.strides_);
  std::swap(size_, temp.size_);
  std::swap(byte_size_, temp.byte_size_);
  std::swap(data_type_, temp.data_type_);
  std::swap(device_type_,temp.device_type_);
  uniq_host_data_.swap(temp.uniq_host_data_);
  std::swap(host_data_, temp.host_data_);
#ifdef USE_CUDA
  uniq_device_data_.swap(temp.uniq_device_data_);
  std::swap(device_data_, temp.device_data_);
#endif
  return *this;
}

NDTensor::~NDTensor() {
//   std::cout<<"deconstruct"<<std::endl;
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
  uniq_host_data_.swap(newTensor.uniq_host_data_);
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
  uniq_host_data_.swap(newTensor.uniq_host_data_);
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
  uniq_host_data_.reset();
  host_data_ = nullptr;
#ifdef USE_CUDA
  uniq_device_data_.reset();
  device_data_ = nullptr;
#endif
}

#ifdef USE_CUDA
NDTensor NDTensor::cpu() {
  if(device_type_==DeviceType::CPU)
    return *this;
  NDTensor t(shapes_,data_type_,DeviceType::CPU);
  cudaMemcpy(t.data<char>(), device_data_, byte_size_, cudaMemcpyDeviceToHost);
  return t;
}
NDTensor NDTensor::cuda() {
  if(device_type_==DeviceType::CUDA)
    return *this;
  NDTensor t(shapes_,data_type_,DeviceType::CUDA);
  cudaMemcpy(t.data<char>(), host_data_, byte_size_, cudaMemcpyHostToDevice);
  return t;
}
#endif

// 访问数据
template <typename T>
T &NDTensor::at(const std::initializer_list<int> &indexs) {
  assert(indexs.size() == shapes_.size());
  assert(device_type_ == DeviceType::CPU);
  assert(sizeof(T) == GetElementSize(data_type_) && "data type mismatch");
  char *ptr = static_cast<char *>(host_data_);
  int i = 0;
  for (auto d : indexs) {
    assert(d < shapes_[i]);  // 检查是否越界，防止踩内存
    ptr += d * strides_[i++] * GetElementSize(data_type_);
  }
  return *((T *)ptr);
}

template <typename T>
const T &NDTensor::at(const std::initializer_list<int> &indexs) const {
  assert(indexs.size() == shapes_.size());
  assert(device_type_ == DeviceType::CPU);
  char *ptr = static_cast<char *>(host_data_);
  int i = 0;
  for (auto d : indexs) {
    ptr += d * strides_[i++] * GetElementSize(data_type_);
  }
  return *ptr;
}

template <typename T>
T *NDTensor::data() {
#ifdef USE_CUDA
  if(device_type_==DeviceType::CUDA)
    return static_cast<T *>(device_data_);
#endif
  return static_cast<T *>(host_data_);
}

template <typename T>
T *NDTensor::data() const {
#ifdef USE_CUDA
  if(device_type_==DeviceType::CUDA)
    return static_cast<T *>(device_data_);
#endif
  return static_cast<T *>(host_data_);
}

template float *NDTensor::data();
template double *NDTensor::data();
template int64_t *NDTensor::data();
template int32_t *NDTensor::data();
template int16_t *NDTensor::data();
template char *NDTensor::data();
template float *NDTensor::data() const;
template double *NDTensor::data() const;
template int64_t *NDTensor::data() const;
template int32_t *NDTensor::data() const;
template int16_t *NDTensor::data() const;
template char *NDTensor::data() const;


std::vector<int> NDTensor::shapes() const {
  return shapes_;
}

std::vector<int> NDTensor::strides() const {
  return strides_;
}

long NDTensor::size() const {
  return size_;
}

long NDTensor::byteSize() const {
  return byte_size_;
}

template <typename T>
T NDTensor::dump2File(const char *filename) const {
  T t=0;
  assert(device_type_ == DeviceType::CPU);
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
  return t;
}

template float NDTensor::dump2File(const char *filename) const;
template double NDTensor::dump2File(const char *filename) const;
template int64_t NDTensor::dump2File(const char *filename) const;
template int32_t NDTensor::dump2File(const char *filename) const;
template int16_t NDTensor::dump2File(const char *filename) const;


int NDTensor::writeFile(const char *filename) const {
  assert(device_type_ == DeviceType::CPU);
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  out.write((char *)host_data_, byte_size_);
  out.close();
  return 0;
}

int NDTensor::readFile(const char *filename) {
  assert(device_type_ == DeviceType::CPU);
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  in.read((char *)host_data_, byte_size_);
  in.close();
  return 0;
}

int NDTensor::readBuff(const std::vector<char> buff) {
  assert(device_type_ == DeviceType::CPU);
  memcpy(host_data_, buff.data(), byte_size_);
  return 0;
}

}  // namespace CUDA_NAMESPACE