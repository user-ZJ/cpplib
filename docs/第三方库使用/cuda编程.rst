CUDA编程
================

https://blog.csdn.net/kyocen/article/details/51424161

.. code-block:: cu
	
	#include <iostream>

	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"

	using namespace std;


	__global__ void add(int a, int b, int *c)
	{
		*c = a + b;
	}


	int main() {
		int c;
		int *dev_c;
		cudaMalloc((void**)&dev_c, sizeof(int));



		add << <1, 1 >> > (2, 7, dev_c);

		cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
		cout << "2+7=" << c << endl;
		cudaFree(dev_c);

		system("pause");
		return 0;
	}


