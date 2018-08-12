#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "parameters.h"

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

inline void get_grid_data(const size_t word_count, unsigned& threads_x, unsigned& grid_x, unsigned& grid_y, unsigned& grid_z)
{
	threads_x = static_cast<unsigned>(BLOCKSIZE < word_count ? BLOCKSIZE : word_count);

	auto num_blocks = word_count % BLOCKSIZE != 0 ? word_count / BLOCKSIZE + 1 : word_count / BLOCKSIZE;

	grid_x = static_cast<unsigned>(GRIDDIM < num_blocks ? GRIDDIM : num_blocks);

	num_blocks = num_blocks % GRIDDIM != 0 ? num_blocks / GRIDDIM + 1 : num_blocks / GRIDDIM;

	grid_y = static_cast<unsigned>(GRIDDIM < num_blocks ? GRIDDIM : num_blocks);

	num_blocks = num_blocks % GRIDDIM != 0 ? num_blocks / GRIDDIM + 1 : num_blocks / GRIDDIM;

	grid_z = static_cast<unsigned>(num_blocks);
}

#define STARTKERNEL(function,name,thread_count,...)														\
{																										\
	unsigned threads_x, grid_x, grid_y, grid_z;															\
	get_grid_data(thread_count, threads_x, grid_x, grid_y, grid_z);										\
																										\
	const dim3 threads(threads_x, 1, 1);																\
	const dim3 blocks(grid_x, grid_y, grid_z);															\
	function kernel_init(blocks, threads) (__VA_ARGS__);												\
	gpuErrchk(cudaGetLastError());																		\
}