#pragma once
#include "parameters.h"
#include <vector>
#include "helper_cuda.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

template <class T>
std::vector<T> create_vector(T* d_pointer, int size)
{
	T* ptr = static_cast<T*>(malloc(sizeof(T) * size));
	checkCudaErrors(cudaMemcpy(ptr, d_pointer, sizeof(T)*size, cudaMemcpyDeviceToHost));
	return std::vector<T>(ptr, ptr + size);
}

inline ullong cpu_hash(std::string s)
{
	int i = 0;
	ullong hash = 0;
	for (; i < CHARSTOHASH; i++)
	{
		const unsigned char c = s[i];
		if (c == '\0')
			break;

		hash *= ALPHABETSIZE;
		hash += c & CHARMASK;
	}

	const ullong mask = s[i] == '\0' ? 0 : 1;

	for (; i < CHARSTOHASH; i++)
		hash *= ALPHABETSIZE;

	hash <<= 1;
	hash |= mask;
	return hash;
}


inline int postfix_len_from_str(std::string str)
{
	const auto c = str.size();
	if (c < CHARSTOHASH)
		return 0;

	return c - CHARSTOHASH + 1;
}

inline void compute_grid_size(uint n, uint block_size, uint &num_blocks, uint &num_threads)
{
	num_threads = block_size < n ? block_size : n;
	num_blocks = (n % num_threads != 0) ? (n / num_threads + 1) : (n / num_threads);
}

template <class T1>
int get_segment_size(T1 max_segment)
{
	int segment_size;
	if (max_segment == 0)
		segment_size = 0;
	else
	{
		segment_size = 32;
		const T1 flag = static_cast<T1>(1) << sizeof(T1) * 8 - 1;
		while ((max_segment&flag) == 0)
		{
			max_segment <<= 1;
			segment_size--;
		}
	}

	return segment_size;
}