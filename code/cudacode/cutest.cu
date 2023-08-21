#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void activeMaskExample() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int active_mask = __activemask();

    if (active_mask & (1 << threadIdx.x)) {
        printf("Thread %d is active.\n", tid);
    }
}




int main() {
    // Define matrix dimensions
    int m = 3; // Number of rows of the first matrix
    int n = 2; // Number of columns of the second matrix
    int k = 3; // Number of columns of the first matrix and rows of the second matrix

    // Allocate memory for input matrices on the host
    float h_A[m * k] = {
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f,
    };

    float h_B[k * n] = {
        1.0f, 1.0f,
        1.0f, 2.0f,
        2.0f, 2.0f
    };

    // Allocate memory for the result matrix on the host
    float h_C[m * n] = {0};

    // Allocate memory for matrices on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define multiplication parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                m, n, k, &alpha,
                d_A, k, d_B, n, &beta,
                d_C, m);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Result matrix:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_C[i * n + j]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}

