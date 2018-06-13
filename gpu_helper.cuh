#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#if __CUDACC__
#define kernel_init(...) <<<__VA_ARGS__>>>
#define only_gpu_assert()

#else
#define __int_as_float(x) (NAN)
#define kernel_init(...)
#define __syncthreads()
#define only_gpu_assert()\
	std::cerr << "There was an attempt to launch a kernel builded without nvcc! Add proper template declaration to any .cu file.\n";\
	assert(false);

const uint3 threadIdx;
const uint3 blockIdx;
const dim3 blockDim;
const dim3 gridDim;
#endif

#define GetGlobalId() (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x\
			         + blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
		{
			system("pause");
			exit(code);
		}
	}
}

template<class T>
__host__ __device__ T my_max(T a, T b)
{
	return (a > b ? a : b);
}

