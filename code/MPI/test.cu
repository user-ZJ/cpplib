#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include "stdio.h"
#include <mpi.h>

using namespace std;

__global__ void VecAdd(float *A,float *B,float *C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

void GetGPUInfo(){
    int device_count;
    cudaGetDeviceCount(&device_count); // GPU个数
    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "totalGlobalMem: " << prop.totalGlobalMem / 1024.0 / 1024 << "MB" << std::endl;
        // computeMode：设备计算模式。
        // computeCapabilityMajor和computeCapabilityMinor：设备的计算能力版本号。
        size_t free_byte, total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);
        std::cout << "Total memory: " << total_byte / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Free memory: " << free_byte / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "used memory: " << (total_byte-free_byte) / (1024.0 * 1024.0) << " MB" << std::endl;
    }
}


int main()
{
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout<<"word_size:"<<world_size<<" world_rank:"<<world_rank<<"\n";

    if(world_rank==0)
        GetGPUInfo();
    

    float *A,*B,*C;
    cudaMalloc(&A, sizeof(float)*10);
    cudaMalloc(&B, sizeof(float)*10);
    cudaMalloc(&C, sizeof(float)*10);
    VecAdd<<<1,10>>>(A,B,C);
    MPI_Finalize();
    return 0;
}