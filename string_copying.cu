#include "gpu_helper.cuh"
#include <thrust/device_vector.h>

__global__ void copy_suffixes_d(const char* words, const int* positions, const size_t word_count, char* suffixes, const int* suffix_positions)
{
	const auto thread_num = GetGlobalId();
	if (thread_num >= word_count)
		return;

	const int suffix_pos = suffix_positions[thread_num];
	const int len = suffix_positions[thread_num + 1] - suffix_pos;
	if (len == 0)
		return;

	if (len == 1)
	{
		suffixes[suffix_pos] = BREAKCHAR;
		return;
	}

	const int position = positions[thread_num] + CHARSTOHASH;

	for (int i = 0; i < len; i++)
		suffixes[suffix_pos + i] = words[position + i];
}

void copy_suffixes(const thrust::device_vector<char>& words, const thrust::device_vector<int>& sorted_positions, const size_t word_count,
                   const thrust::device_vector<int>& suffix_positions, thrust::device_vector<char>& suffixes)
{
	STARTKERNEL(copy_suffixes_d, word_count, words.data().get(), sorted_positions.data().get(),
		word_count, suffixes.data().get(), suffix_positions.data().get());
}

__global__ void reposition_strings_d(char* d_word_array_in, char* d_word_array_out, int* d_position_in,
	int* d_position_out, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const int position_in = d_position_in[thread_num];
	const int position_out = d_position_out[thread_num];

	int i = 0;
	char c;
	do
	{
		c = d_word_array_in[position_in + i];
		d_word_array_out[position_out + i] = c;
		i++;
	} while (c != BREAKCHAR);
}
