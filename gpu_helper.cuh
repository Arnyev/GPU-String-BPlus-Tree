#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "parameters.h"

#if __NVCC__
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

#define GetGlobalId() ((blockIdx.x + blockIdx.y * static_cast<size_t>(gridDim.x) + static_cast<size_t>(gridDim.x) * gridDim.y * blockIdx.z)\
 * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x+ threadIdx.x)

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

#define MEASURETIME(function, name)													\
{																					\
	cudaEvent_t start;																\
	cudaEvent_t stop;																\
	float milliseconds = 0;															\
	if (WRITETIME)																	\
	{																				\
		checkCudaErrors(cudaEventCreate(&start));									\
		checkCudaErrors(cudaEventCreate(&stop));									\
		checkCudaErrors(cudaDeviceSynchronize());									\
		checkCudaErrors(cudaEventRecord(start));									\
	}																				\
    (function);																		\
	if (WRITETIME)																	\
	{																				\
		checkCudaErrors(cudaEventRecord(stop));										\
		checkCudaErrors(cudaEventSynchronize(stop));								\
		checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));			\
		std::cout << name << " took " << milliseconds << " ms" << std::endl;		\
		checkCudaErrors(cudaEventDestroy(start));									\
		checkCudaErrors(cudaEventDestroy(stop));									\
	}																				\
}

#define MEASURETIMEKERNEL(function,name,threadCount,...)							\
{																					\
	cudaEvent_t start;																\
	cudaEvent_t stop;																\
	float milliseconds = 0;															\
	if (WRITETIME)																	\
	{																				\
		checkCudaErrors(cudaEventCreate(&start));									\
		checkCudaErrors(cudaEventCreate(&stop));									\
		checkCudaErrors(cudaDeviceSynchronize());									\
		checkCudaErrors(cudaEventRecord(start));									\
	}																				\
	uint num_threads, num_blocks;													\
	num_threads = BLOCKSIZE < threadCount ? BLOCKSIZE : threadCount;				\
	num_blocks = (threadCount % num_threads != 0) ?									\
		(threadCount / num_threads + 1) : (threadCount / num_threads);				\
	function kernel_init(num_blocks, num_threads) (__VA_ARGS__);					\
	getLastCudaError(name " failed.");												\
	if (WRITETIME)																	\
	{																				\
		checkCudaErrors(cudaEventRecord(stop));										\
		checkCudaErrors(cudaEventSynchronize(stop));								\
		checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));			\
		std::cout << name << " took " << milliseconds << " ms" << std::endl;		\
		checkCudaErrors(cudaEventDestroy(start));									\
		checkCudaErrors(cudaEventDestroy(stop));									\
	}																				\
}

#define STARTKERNEL(function,name,thread_count,...)														\
{																										\
	const auto threads_x = static_cast<unsigned>(BLOCKSIZE < thread_count ? BLOCKSIZE : thread_count);	\
	auto num_blocks = thread_count % threads_x != 0														\
		? thread_count / threads_x + 1																	\
		: thread_count / threads_x;																		\
																										\
	const auto grid_x = static_cast<unsigned>(GRIDDIM < num_blocks ? GRIDDIM : num_blocks);				\
	num_blocks = num_blocks / GRIDDIM == 0 ? 1 : num_blocks / GRIDDIM;									\
	const auto grid_y = static_cast<unsigned>(GRIDDIM < num_blocks ? GRIDDIM : num_blocks);				\
	num_blocks = num_blocks / GRIDDIM == 0 ? 1 : num_blocks / GRIDDIM;									\
	const auto grid_z = static_cast<unsigned>(num_blocks);												\
																										\
	const dim3 threads(threads_x, 1, 1);																\
	const dim3 blocks(grid_x, grid_y, grid_z);															\
	function kernel_init(num_blocks, num_threads) (__VA_ARGS__);										\
	getLastCudaError(name " failed.");																	\
}