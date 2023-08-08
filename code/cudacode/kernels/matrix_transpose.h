#ifndef _MATRIX_TRANSPOSE_H_
#define _MATRIX_TRANSPOSE_H_

// @reduction_kernel.cu
int matrix_transpose(float *d_out, float *d_in,int N);

int global_matrix_transpose(float *d_out, float *d_in,int N);


#endif // _MATRIX_TRANSPOSE_H_
