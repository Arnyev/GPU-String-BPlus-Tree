#pragma once
#include <chrono>
#include "cuda_runtime.h"
#include "helper_cuda.h"

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

		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		checkCudaErrors(cudaEventRecord(start));

		std::forward<decltype(func)>(func)(std::forward<Args>(args)...);

		checkCudaErrors(cudaEventRecord(stop));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
		checkCudaErrors(cudaEventDestroy(start));
		checkCudaErrors(cudaEventDestroy(stop));

		return milliseconds * 1000;
	}
};

template <int B>
int needed_nodes(int elemNum)
{
	if (elemNum < B)
		return 1;
	int pages = elemNum * 2 / B;
	elemNum = pages;
	while (elemNum > B + 1)
	{
		elemNum = elemNum / (B / 2 + 1);
		pages += elemNum;
	}
	pages += 1;
	return pages;
}

template<class T>
std::vector<T> from_vector_host(const thrust::host_vector<T>& host_vector)
{
	std::vector<T> result(host_vector.size());
	memcpy(result.data(), host_vector.data(), host_vector.size() * sizeof(T));
	return result;
}

template<>
inline std::vector<bool> from_vector_host(const thrust::host_vector<bool>& host_vector)
{
	const auto vals = new bool[host_vector.size()];
	memcpy(vals, host_vector.data(), host_vector.size() * sizeof(bool));
	auto result = std::vector<bool>(vals, vals + host_vector.size());
	delete[]vals;
	return result;
}

template<class T>
std::vector<T> from_vector_dev(const thrust::device_vector<T>& device_vector)
{
	std::vector<T> result(device_vector.size());
	checkCudaErrors(cudaMemcpy(result.data(), device_vector.data().get(), sizeof(T)*device_vector.size(), cudaMemcpyDeviceToHost));
	return result;
}

template<>
inline std::vector<bool> from_vector_dev(const thrust::device_vector<bool>& device_vector)
{
	const auto vals = new bool[device_vector.size()];
	checkCudaErrors(cudaMemcpy(vals, device_vector.data().get(), sizeof(bool)*device_vector.size(), cudaMemcpyDeviceToHost));
	auto result = std::vector<bool>(vals, vals + device_vector.size());
	delete[]vals;
	return result;
}
