#ifndef _HELPER_H_
#define _HELPER_H_

#include <cublas_v2.h>
#include <cudnn.h>

// #include <helper_cuda.h>
#include <curand.h>

namespace CUDA_NAMESPACE {
#define BLOCK_DIM_1D 512
#define BLOCK_DIM 16

struct StreamDeleter {
  void operator()(cudaStream_t *stream) {
    if (stream) {
      cudaStreamDestroy(*stream);
      delete stream;
    }
  }
};

template <typename T>
struct CuMemDeleter {
  void operator()(T *p) noexcept {
    cudaFree(p);
  }
};

template <typename T, template <typename> class DeleterType = CuMemDeleter>
using UniqPtr = std::unique_ptr<T, DeleterType<T>>;

/* CUDA API error return checker */
#ifndef checkCudaErrors
#define checkCudaErrors(err)                                                                                           \
  {                                                                                                                    \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                                                            \
      fprintf(stderr, "%d\n", cudaSuccess);                                                                            \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }
#endif

template <typename T>
inline UniqPtr<T, CuMemDeleter> mallocCudaMem(size_t nbElems) {
  T *ptr = nullptr;
  checkCudaErrors(cudaMalloc((void **)&ptr, sizeof(T) * nbElems));
  return UniqPtr<T, CuMemDeleter>{ptr};
}

inline std::unique_ptr<cudaStream_t, StreamDeleter> makeCudaStream(int flags = cudaStreamNonBlocking) {
  // cudaStream_t stream;
  // checkCudaErrors(cudaStreamCreateWithFlags(&stream, flags));
  // return std::unique_ptr<cudaStream_t, StreamDeleter>{ stream };
  std::unique_ptr<cudaStream_t, StreamDeleter> pStream(new cudaStream_t);
  if (cudaStreamCreateWithFlags(pStream.get(), flags) != cudaSuccess) { pStream.reset(nullptr); }
  return pStream;
}

static const char *_cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

#define checkCublasErrors(err)                                                                                         \
  {                                                                                                                    \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                                \
      fprintf(stderr, "checkCublasErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                   \
              _cublasGetErrorEnum(err), __FILE__, __LINE__);                                                           \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

#define checkCudnnErrors(err)                                                                                          \
  {                                                                                                                    \
    if (err != CUDNN_STATUS_SUCCESS) {                                                                                 \
      fprintf(stderr, "checkCudnnErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                    \
              cudnnGetErrorString(err), __FILE__, __LINE__);                                                           \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

// cuRAND API errors
static const char *_curandGetErrorEnum(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";

  case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";

  case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";

  case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";

  case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";

  case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";

  case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

  case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";

  case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";

  case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";

  case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";

  case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

#define checkCurandErrors(err)                                                                                         \
  {                                                                                                                    \
    if (err != CURAND_STATUS_SUCCESS) {                                                                                \
      fprintf(stderr, "checkCurandErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                   \
              _curandGetErrorEnum(err), __FILE__, __LINE__);                                                           \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

// container for cuda resources
class CudaContext {
 public:
  CudaContext() {
    cublasCreate(&_cublas_handle);
    checkCudaErrors(cudaGetLastError());
    checkCudnnErrors(cudnnCreate(&_cudnn_handle));
  }
  ~CudaContext() {
    cublasDestroy(_cublas_handle);
    checkCudnnErrors(cudnnDestroy(_cudnn_handle));
  }

  cublasHandle_t cublas() {
    // std::cout << "Get cublas request" << std::endl; getchar();
    return _cublas_handle;
  };
  cudnnHandle_t cudnn() {
    return _cudnn_handle;
  };

  const float one = 1.f;
  const float zero = 0.f;
  const float minus_one = -1.f;

 private:
  cublasHandle_t _cublas_handle;
  cudnnHandle_t _cudnn_handle;
};


class TensorDescriptorRAII {
 public:
  // tensor结构指针
  cudnnTensorDescriptor_t tensor_desc_;
  TensorDescriptorRAII() {
    cudnnCreateTensorDescriptor(&tensor_desc_);
  }
  ~TensorDescriptorRAII() {
    if (tensor_desc_) checkCudnnErrors(cudnnDestroyTensorDescriptor(tensor_desc_));
  }
  // 深拷贝
  TensorDescriptorRAII(const TensorDescriptorRAII &t) {
    if (!tensor_desc_) cudnnCreateTensorDescriptor(&tensor_desc_);
  }
  TensorDescriptorRAII &operator=(const TensorDescriptorRAII &t) {
    if (this == &t) return *this;
    if (!tensor_desc_) cudnnCreateTensorDescriptor(&tensor_desc_);
    return *this;
  }
  TensorDescriptorRAII(TensorDescriptorRAII &&t) {
    tensor_desc_ = t.tensor_desc_;
    t.tensor_desc_ = nullptr;
  }
  TensorDescriptorRAII &operator=(TensorDescriptorRAII &&t) {
    tensor_desc_ = t.tensor_desc_;
    t.tensor_desc_ = nullptr;
    return *this;
  }
};

inline TensorDescriptorRAII createTensorDesc(const std::vector<int> &shape) {
  TensorDescriptorRAII tensor_desc_raii;
  if (shape.size() <= 4) {
    std::array<int, 4> nchw{1, 1, 1, 1};
    for (int i = 0; i < shape.size(); i++)
      nchw[i] = shape[i];
    cudnnSetTensor4dDescriptor(tensor_desc_raii.tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, nchw[0], nchw[1],
                               nchw[2], nchw[3]);
  }
  return tensor_desc_raii;
}

}  // namespace CUDA_NAMESPACE

#endif  // _HELPER_H_