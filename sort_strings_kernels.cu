#include "parameters.h"
#include <device_launch_parameters.h>
#include "sort_helpers.cuh"
#include "gpu_helper.cuh"
#include <helper_cuda.h>
#include <thrust/device_ptr.h>

using namespace thrust;

__global__ void reposition_strings_d(unsigned char* d_word_array_in, unsigned char* d_word_array_out,
	int* d_position_in, int* d_position_out, const int word_count)
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

__global__ void flag_different_than_last_d(ullong* keys, int* flags, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	flags[thread_num] = thread_num == 0 || keys[thread_num] != keys[thread_num - 1] ? 1 : 0;
}

__global__ void create_hashes_with_seg_d(uchar* words, int* word_positions, int* segments, ullong* keys,
	const int offset, const int chars_to_hash, const int seg_shift,
	const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const int position = word_positions[thread_num] + offset;

	keys[thread_num] = static_cast<ullong>(segments[thread_num]) << seg_shift | get_hash(words, chars_to_hash, position);
}

__global__ void mark_singletons_d(ullong* keys, int* flags, int* destinations, int* output, int* word_positions,
	const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const ullong key = keys[thread_num];
	const int word_position = word_positions[thread_num];
	const bool finished = (key & 1ULL) == 0ULL;
	const int index_output = destinations[thread_num];

	if (thread_num == 0)
	{
		if (finished || key != keys[thread_num + 1])
		{
			output[index_output] = word_position;
			flags[thread_num] = 0;
		}
		else
			flags[thread_num] = 1;

		return;
	}

	const auto key_last = keys[thread_num - 1];

	if (thread_num == word_count - 1)
	{
		if (key != key_last)
		{
			output[index_output] = word_position;
			flags[thread_num] = 0;
		}
		else if (finished)
		{
			output[index_output] = -1;
			flags[thread_num] = 0;
		}
		else
			flags[thread_num] = 1;

		return;
	}

	const ullong key_next = keys[thread_num + 1];

	if (key != key_last && (finished || key != key_next))
	{
		output[index_output] = word_position;
		flags[thread_num] = 0;
	}
	else if (key == key_last && finished)
	{
		output[index_output] = -1;
		flags[thread_num] = 0;
	}
	else
		flags[thread_num] = 1;
}

__global__ void create_consecutive_numbers_d(int* numbers, const int max_number)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= max_number)
		return;

	numbers[thread_num] = thread_num;
}

__global__ void compute_postfix_lengths_d(uchar* words, int* positions, const int word_count, int* lengths)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	int my_position = positions[thread_num];
	if (my_position == -1)
		return;

	int length = 0;
	uchar c;
	for (int i = 1; i < CHARSTOHASH; i++)
	{
		c = words[my_position + i];
		if (c == BREAKCHAR)
			return;
	}

	my_position = my_position + CHARSTOHASH;
	while (true)
	{
		c = words[my_position];

		if (c == BREAKCHAR)
			break;
		my_position++;
		length++;
	}

	lengths[thread_num] = length + 1;
}

__global__ void copy_suffixes_d(uchar* words, int* positions, const int word_count, uchar* suffixes,
	int* suffix_positions)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const int suffix_pos = suffix_positions[thread_num];
	const int len = suffix_positions[thread_num + 1] - suffix_pos;
	if (len == 0)
		return;

	const int position = positions[thread_num] + CHARSTOHASH;

	for (int i = 0; i < len; i++)
		suffixes[suffix_pos + i] = words[position + i];
}

void copy_suffixes(unsigned char* d_word_array, int* d_sorted_positions, int word_count, const device_ptr<int> suffix_positions, device_ptr<unsigned char> suffixes)
{
	MEASURETIMEKERNEL(copy_suffixes_d, "Copying suffixes", word_count, d_word_array, d_sorted_positions, word_count, suffixes.get(), suffix_positions.get());
}

void flags_different_than_last(ullong* d_keys, int* d_flags, int current_count)
{
	MEASURETIMEKERNEL(flag_different_than_last_d, "Flags different than last", current_count, d_keys, d_flags, current_count);
}

void create_consecutive_numbers(const int word_count, device_ptr<int> destinations)
{
	MEASURETIMEKERNEL(create_consecutive_numbers_d, "Create consecutive numbers", word_count, destinations.get(), word_count);
}

void create_hashes_with_seg(int* d_positions, unsigned char* d_chars, device_ptr<unsigned long long> keys, device_ptr<int> helper, int offset, int segment_size, int current_count, const int seg_chars)
{
	MEASURETIMEKERNEL(create_hashes_with_seg_d, "Create hashes", current_count, d_chars, d_positions, helper.get(), keys.get(),
		offset, CHARSTOHASH - seg_chars, KEYBITS - segment_size, current_count);
}

void mark_singletons(int* d_positions, device_ptr<unsigned long long> keys, device_ptr<int> destinations, device_ptr<int> helper, device_ptr<int> output, int current_count)
{
	MEASURETIMEKERNEL(mark_singletons_d, "Marking singletons", current_count, keys.get(), helper.get(), destinations.get(), output.
		get(), d_positions, current_count);
}
