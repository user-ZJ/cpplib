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

class Dropout {
 public:
  cudnnHandle_t cudnn;
  cudnnDropoutDescriptor_t dropout_descriptor;
  size_t dropout_state_size;
  size_t dropout_reserve_size;

  cudnnTensorDescriptor_t dropout_in_out_descriptor;

  float dropRate;
  float *ref_input{nullptr};
  float *d_dropout_out{nullptr};
  float *d_dx_dropout{nullptr};
  void *states;
  void *dropout_reserve_space;
  int batchSize, features, imgH, imgW;
  int in_out_bytes;

  Dropout(float dropR, int batchS, int feat, int imageH, int imageW) :
    dropRate(dropR), batchSize(batchS), features(feat), imgH(imageH), imgW(imageW) {
    in_out_bytes = sizeof(float) * batchSize * features * imgH * imgW;

    checkCUDNN(cudnnCreate(&cudnn));
    checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&dropout_in_out_descriptor));

    checkCUDNN(cudnnSetTensor4dDescriptor(dropout_in_out_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize,
                                          features, imgH, imgW));

    checkCUDNN(cudnnDropoutGetStatesSize(cudnn, &dropout_state_size));

    checkCUDNN(cudnnDropoutGetReserveSpaceSize(dropout_in_out_descriptor, &dropout_reserve_size));

    // Allocate memory for states and reserve space
    cudaMalloc(&states, dropout_state_size);
    cudaMalloc(&dropout_reserve_space, dropout_reserve_size);

    checkCUDNN(cudnnSetDropoutDescriptor(dropout_descriptor, cudnn, dropRate, states, dropout_state_size,
                                         /*Seed*/ time(NULL)));

    cudaMalloc(&d_dropout_out, in_out_bytes);
    cudaMalloc(&d_dx_dropout, in_out_bytes);
  };

  float *Forward(float *d_input) {
    ref_input = d_input;

    printf("Input \n");
    GPU_PrintArr3D(d_input, imgH, imgW, batchSize * features);
    checkCUDNN(cudnnDropoutForward(cudnn, dropout_descriptor, dropout_in_out_descriptor, ref_input,
                                   dropout_in_out_descriptor, d_dropout_out, dropout_reserve_space,
                                   dropout_reserve_size));

    printf("Dropout \n");
    GPU_PrintArr3D(d_dropout_out, imgH, imgW, batchSize * features);

    return d_dropout_out;
  }

  float *Backward(float *d_in_grads) {
    checkCUDNN(cudnnDropoutBackward(cudnn, dropout_descriptor, dropout_in_out_descriptor, d_in_grads,
                                    dropout_in_out_descriptor, d_dx_dropout, dropout_reserve_space,
                                    dropout_reserve_size));

    printf("Dropout Grad\n");
    GPU_PrintArr3D(d_dx_dropout, imgH, imgW, features * batchSize);
    return d_dx_dropout;
  }
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