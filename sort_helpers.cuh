#pragma once

#include "parameters.h"
#include <crt/host_defines.h>

__device__ __host__ __inline__ ullong get_hash(uchar* words, const int chars_to_hash, const int my_position)
{
	uchar last_bit = 1;
	uchar char_mask = CHARMASK;

	ullong hash = 0;

	for (int i = 0; i < chars_to_hash; i++)
	{
		const unsigned char c = words[i + my_position];
		if (c == BREAKCHAR)
		{
			char_mask = 0;
			last_bit = 0;
		}
		hash *= ALPHABETSIZE;
		hash += c & char_mask;
	}
	if (words[chars_to_hash + my_position] == BREAKCHAR)
		last_bit = 0;

	return hash << 1 | last_bit;
}

inline void compute_grid_size(uint n, uint block_size, uint &num_blocks, uint &num_threads)
{
	num_threads = block_size < n ? block_size : n;
	num_blocks = (n % num_threads != 0) ? (n / num_threads + 1) : (n / num_threads);
}
