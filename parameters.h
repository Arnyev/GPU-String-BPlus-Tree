#pragma once

#include <vector>
#include <thrust/device_vector.h>
#include <chrono>
#include <helper_cuda.h>

typedef unsigned char uchar;
typedef unsigned long long ullong;
typedef unsigned int uint;

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define BREAKCHAR '\0'
#define SPLITTER ' '
#define FILEPATH "book.txt"
#define RANDSTRMINLEN 1
#define RANDSTRMAXLEN 100
#define RANDSTRCOUNT 1000000ULL
#define RANDCHARSET "abcdef"
#define RANDCHARSCOUNT (sizeof(RANDCHARSET)-1)
#define CHARSTOHASH 13
#define ALPHABETSIZE 27
#define BLOCKSIZE 256
#define GRIDDIM 2048
#define KEYBITS 64
#define CHARBITS 5
#define CHARMASK ~static_cast<uchar>(3 << 5);
#define WRITETIME 1
#define TOLOWERMASK (1<<5)

struct sorting_output_gpu
{
	thrust::device_vector<ullong> hashes;
	thrust::device_vector<int> positions;
	thrust::device_vector<uchar> suffixes;
};

struct sorting_output_cpu
{
	std::vector<ullong> hashes;
	std::vector<int> positions;
	std::vector<uchar> suffixes;
};

struct measure
{
	template<typename F, typename ...Args>
	static std::chrono::microseconds::rep execution(F&& func, Args&&... args)
	{
		const auto start = std::chrono::high_resolution_clock::now();
		std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::high_resolution_clock::now() - start);
		return duration.count();
	}

	template<typename F, typename ...Args>
	static float execution_gpu(F&& func, Args&&... args)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		float milliseconds = 0;

		if (WRITETIME)
		{
			checkCudaErrors(cudaEventCreate(&start));
			checkCudaErrors(cudaEventCreate(&stop));
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaEventRecord(start));
		}
		std::forward<decltype(func)>(func)(std::forward<Args>(args)...);

		if (WRITETIME)
		{
			checkCudaErrors(cudaEventRecord(stop));
			checkCudaErrors(cudaEventSynchronize(stop));
			checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
			checkCudaErrors(cudaEventDestroy(start));
			checkCudaErrors(cudaEventDestroy(stop));
		}
		return milliseconds * 1000;
	}
};

template<class T>
std::vector<T> from_vector_host(const thrust::host_vector<T>& host_vector)
{
	std::vector<T> result(host_vector.size());
	for (int i = 0; i < host_vector.size(); i++)
		result[i] = host_vector[i];

	return result;
}

template<class T>
std::vector<T> from_vector_dev(const thrust::device_vector<T>& device_vector)
{
	thrust::host_vector<T> host_vector(device_vector);
	return from_vector_host(host_vector);
}
