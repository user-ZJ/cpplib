#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

#define checkCUDNN(expression)                                                                                         \
  {                                                                                                                    \
    cudnnStatus_t status = (expression);                                                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                                                              \
      std::cerr << "cuDNN Error in File " << __FILE__ << " Error on line " << __LINE__ << ": "                         \
                << cudnnGetErrorString(status) << std::endl;                                                           \
      std::exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                                  \
  }

void printArr3D(float *arr, int arrH, int arrW, int batchSize) {
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < arrH; j++) {
      for (int k = 0; k < arrW; k++) {
        printf("%f ", arr[i * arrH * arrW + j * arrW + k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

void GPU_PrintArr3D(float *d_arr, int arrH, int arrW, int batchSize) {
  float *h_arr;
  h_arr = (float *)malloc(sizeof(float) * arrH * arrW * batchSize);
  cudaMemcpy(h_arr, d_arr, sizeof(float) * arrH * arrW * batchSize, cudaMemcpyDeviceToHost);
  printArr3D(h_arr, arrH, arrW, batchSize);
}

class Activation {
 public:
  Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f) {
    name_ = name;
    act_mode_ = mode;
    act_coef_ = coef;

    cudnnCreateActivationDescriptor(&act_desc_);
    cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
  }
  virtual ~Activation() {
    cudnnDestroyActivationDescriptor(act_desc_);
  }

  void fwd_initialize(Blob<float> *input) {
    if (input_ == nullptr || batch_size_ != input->n()) {
      input_ = input;
      input_desc_ = input->tensor();
      batch_size_ = input->n();

      if (output_ == nullptr)
        output_ = new Blob<float>(input->shape());
      else
        output_->reset(input->shape());

      output_desc_ = output_->tensor();
    }
  }

  virtual Blob<float> *forward(Blob<float> *input) {
    cudnnActivationForward(cuda_->cudnn(), act_desc_, &cuda_->one, input_desc_, input->cuda(), &cuda_->zero,
                           output_desc_, output_->cuda());

    return output_;
  }

  void bwd_initialize(Blob<float> *grad_output) {
    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
      grad_output_ = grad_output;

      if (grad_input_ == nullptr)
        grad_input_ = new Blob<float>(input_->shape());
      else
        grad_input_->reset(input_->shape());
    }
  }
  virtual Blob<float> *backward(Blob<float> *grad_input) {
    cudnnActivationBackward(cuda_->cudnn(), act_desc_, &cuda_->one, output_desc_, output_->cuda(), output_desc_,
                            grad_output->cuda(), input_desc_, input_->cuda(), &cuda_->zero, input_desc_,
                            grad_input_->cuda());

    return grad_input_;
  }

 private:
  cudnnActivationDescriptor_t act_desc_;
  cudnnActivationMode_t act_mode_;
  float act_coef_;
};

int main() {
  int batchSize = 1;
  int imgH = 5;
  int imgW = 5;
  int inC = 3;
  float dropRate = 0.5;

  int n = batchSize * imgH * imgW * inC;
  int in_out_bytes = n * sizeof(float);

  float *h_input, *d_input;
  float *d_output;
  float *h_grads, *d_grads;
  h_input = (float *)malloc(in_out_bytes);
  h_grads = (float *)malloc(in_out_bytes);
  cudaMalloc(&d_input, in_out_bytes);
  cudaMalloc(&d_grads, in_out_bytes);

  for (int i = 0; i < n; i++)
    h_input[i] = 1.0;

  Dropout dLayer = Dropout(dropRate, batchSize, inC, imgH, imgW);

  cudaMemcpy(d_input, h_input, in_out_bytes, cudaMemcpyHostToHost);

  d_output = dLayer.Forward(d_input);

  cudaMemcpy(h_grads, d_output, in_out_bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < batchSize * imgH * imgW * inC; i++)
    h_grads[i] *= 2;

  cudaMemcpy(d_grads, h_grads, in_out_bytes, cudaMemcpyHostToDevice);
  d_output = dLayer.Backward(d_grads);
}