#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

using namespace std;

int main()
{
    // Create a cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set matrix dimensions and data type
    int m = 2; // number of rows
    int n = 3; // number of columns
    int k = 2; // number of columns in weight matrix
    float alpha = 1.0; // scaling factor for input matrix
    float beta = 0.0; // scaling factor for output matrix

    // Define input and weight matrices
    float* A = new float[m * k]; // input matrix
    float* B = new float[k * n]; // weight matrix

    // Initialize input matrix and weight matrix
    for (int i = 0; i < m * k; i++)
    {
        A[i] = i+1;
    }
    for (int i = 0; i < k * n; i++)
    {
        B[i] = i+1;
    }

    // Define output matrix
    float* C = new float[m * n]; // output matrix

    float *dA,*dB,*dC;
    cudaMalloc(&dA, sizeof(float)*m*k);
    cudaMalloc(&dB, sizeof(float)*k*n);
    cudaMalloc(&dC, sizeof(float)*m*n);

    cudaMemcpy(dA, A, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float)*k*n, cudaMemcpyHostToDevice);

    // Perform matrix multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, n, dA, k, &beta, dC, n);

    cudaMemcpy(C, dC, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    // Print output matrix
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << C[i * n + j] << " ";
        }
        cout << endl;
    }

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    cublasDestroy(handle);

    return 0;
}