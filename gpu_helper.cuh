#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#if __NVCC__
#define kernel_init(...) <<<__VA_ARGS__>>>
#else
#define __int_as_float(x) (NAN)
#define kernel_init(...)
#define __syncthreads()
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

