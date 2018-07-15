#include "device_launch_parameters.h"
#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "parameters.h"
#include "functions.h"
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/device_malloc.h>
#include <thrust/transform_scan.h>
#include <thrust/device_free.h>
#include "DeviceFunctions.cuh"

using namespace thrust;
using namespace std;

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

template <class T1>
__global__ void mark_singletons_d(ullong* keys, T1* flags, int* destinations, int* output, int* word_positions,
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

template <class T1,class T2>
__global__ void flag_different_than_last_d(T1* keys, T2* segments, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	segments[thread_num] = thread_num == 0 || keys[thread_num] != keys[thread_num - 1] ? 1 : 0;
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
	while(true)
	{
		c = words[my_position];

		if (c == BREAKCHAR)
			break;
		my_position++;
		length++;
	}

	lengths[thread_num] = length + 1;
}

__global__ void copy_suffixes(uchar* words, int* positions, const int word_count, uchar* suffixes,
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

template<class T>
struct equal_to_minus_one : thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T x) const { return x == -1; }
};

template<class T>
struct equal_to_zero : thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T x) const { return x == 0; }
};

struct hash_functor: thrust::unary_function<int, ullong>
{
	uchar* words;

	explicit hash_functor(uchar* words): words(words) {	}

	__host__ __device__ ullong operator()(const int position) const 
	{
		if (position == -1)
			return 0ULL;

		return get_hash(words, CHARSTOHASH, position);
	}
};

struct compute_postfix_length_functor : thrust::unary_function<int, int>
{
	uchar* words;

	__device__  int operator()(int my_position) const
	{
		if (my_position == -1)
			return 0;

		int length = 0;
		uchar c;
		for (int i = 1; i < CHARSTOHASH; i++)
		{
			c = words[my_position + i];
			if (c == BREAKCHAR)
				return 0;
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

		return length + 1;
	}
};

void create_hashes(uchar* d_word_array, const device_ptr<int> sorted_positions, const device_ptr<int> positions_end,
                   const device_ptr<ullong> hashes)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;

	if (WRITETIME)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaDeviceSynchronize();
		cudaEventRecord(start);
	}

	transform(sorted_positions, positions_end, hashes, hash_functor(d_word_array));

	if (WRITETIME)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Hashes simple took " << milliseconds << " milliseconds" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

sorting_output create_output(unsigned char* d_word_array, int* d_sorted_positions, int word_count)
{
	const device_ptr<int> sorted_positions(d_sorted_positions);

	const auto positions_end = remove_if(sorted_positions, device_ptr<int>(d_sorted_positions + word_count), equal_to_minus_one<int>());

	word_count = positions_end - sorted_positions;

	uint num_threads, num_blocks;
	compute_grid_size(word_count, BLOCKSIZE, num_blocks, num_threads);

	compute_postfix_length_functor postfix_functor;
	postfix_functor.words = d_word_array;

	const auto suffix_positions = device_malloc<int>(word_count + 1);
	transform_exclusive_scan(sorted_positions, positions_end + 1, suffix_positions, postfix_functor, 0, thrust::plus<int>());

	int output_size;
	checkCudaErrors(cudaMemcpy(&output_size, suffix_positions.get() + word_count, sizeof(int), cudaMemcpyDeviceToHost));

	auto suffixes = device_malloc<uchar>(output_size);
	copy_suffixes << <num_blocks, num_threads >> > (d_word_array, d_sorted_positions, word_count, suffixes.get(), suffix_positions.get());

	const auto hashes = device_malloc<ullong>(word_count);
	create_hashes(d_word_array, sorted_positions, positions_end, hashes);

	const auto hashes_end = unique_by_key(hashes, hashes + word_count, suffix_positions);
	const int hashes_count = hashes_end.first - hashes;

	return { hashes.get(), suffix_positions.get(), suffixes.get(), hashes_count, output_size };
}

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

template <class T1>
void mark_singletons(ullong* d_keys, int* d_word_positions, int* d_destinations, T1* d_flags, int* d_output, int current_count)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;

	if (WRITETIME)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaDeviceSynchronize();
		cudaEventRecord(start);
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(current_count, BLOCKSIZE, num_blocks, num_threads);
	mark_singletons_d << <num_blocks, num_threads >> > (d_keys, d_flags, d_destinations, d_output, d_word_positions, current_count);
	getLastCudaError("Singletons failed.");

	if (WRITETIME)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Singletons took " << milliseconds << " milliseconds" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

template <class T>
void flags_different_than_last(ullong* d_keys, T* d_flags, int current_count)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;

	if (WRITETIME)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaDeviceSynchronize();
		cudaEventRecord(start);
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(current_count, BLOCKSIZE, num_blocks, num_threads);

	flag_different_than_last_d << <num_blocks, num_threads >> > (d_keys, d_flags, current_count);
	getLastCudaError("Finding different flags failed.");

	if (WRITETIME)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Finding different than last took " << milliseconds << " milliseconds" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

int compute_segment_size(int* d_helper, const int current_count)
{
	int max_segment;
	checkCudaErrors(cudaMemcpy(&max_segment, d_helper + current_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
	return get_segment_size(max_segment);
}

void create_hashes_with_seg(ullong* d_keys, unsigned char* d_word_array, int* d_word_positions, int* d_helper,
                            const int offset, const int segment_size, const int current_count, const int seg_chars)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;

	if (WRITETIME)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaDeviceSynchronize();
		cudaEventRecord(start);
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(current_count, BLOCKSIZE, num_blocks, num_threads);
	create_hashes_with_seg_d << <num_blocks, num_threads >> > (d_word_array, d_word_positions, d_helper, d_keys, offset,
	                                                           CHARSTOHASH - seg_chars, KEYBITS - segment_size, current_count);
	getLastCudaError("Creating hashes failed.");

	if (WRITETIME)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Create hashes took " << milliseconds << " milliseconds" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

void create_consecutive_numbers(const int word_count, int* d_destinations)
{	
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;

	if (WRITETIME)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaDeviceSynchronize();
		cudaEventRecord(start);
	}

	uint num_threads;
	uint num_blocks;
	compute_grid_size(word_count, BLOCKSIZE, num_blocks, num_threads);
	create_consecutive_numbers_d << <num_blocks, num_threads >> > (d_destinations, word_count);
	getLastCudaError("Consecutive numbers failed.");

	if (WRITETIME)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Consecutive numbers took " << milliseconds << " milliseconds" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

void sort_wrapper(const device_ptr<ullong> keys, const device_ptr<int> positions, const int current_count)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;

	if (WRITETIME)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaDeviceSynchronize();
		cudaEventRecord(start);
	}

	sort_by_key(keys, keys + current_count, positions);

	if (WRITETIME)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Sorting took " << milliseconds << " milliseconds" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

int remove_handled(const device_ptr<int> positions, const device_ptr<ullong> keys, const device_ptr<int> destinations,
	const device_ptr<int> helper, const int current_count)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;

	if (WRITETIME)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaDeviceSynchronize();
		cudaEventRecord(start);
	}

	const auto iter_start = make_zip_iterator(thrust::make_tuple(keys, positions, destinations));

	const auto iter_end = make_zip_iterator(
		thrust::make_tuple(keys + current_count, positions + current_count, destinations + current_count));

	const auto new_end = remove_if(iter_start, iter_end, helper, equal_to_zero<uchar>());

	if (WRITETIME)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Removing handled took " << milliseconds << " milliseconds" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	return new_end - iter_start;
}

int* get_sorted_positions(int* d_positions, const int word_count, unsigned char* d_chars)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;
	cudaEventRecord(start);

	auto keys = device_malloc<ullong>(word_count);
	auto destinations = device_malloc<int>(word_count);
	auto helper = device_malloc<int>(word_count);
	auto output = device_malloc<int>(word_count);

	create_consecutive_numbers(word_count, destinations.get());

	int offset = 0;
	int segment_size = 0;
	int current_count = word_count;

	while (true)
	{
		const int seg_chars = ceil(static_cast<double>(segment_size) / CHARBITS);
		create_hashes_with_seg(keys.get(), d_chars, d_positions, helper.get(), offset, segment_size, current_count, seg_chars);
		offset += CHARSTOHASH - seg_chars;

		sort_wrapper(keys, device_ptr<int>(d_positions), current_count);

		mark_singletons(keys.get(), d_positions, destinations.get(), helper.get(), output.get(), current_count);

		current_count = remove_handled(device_ptr<int>(d_positions), keys, destinations, helper, current_count);
		if (current_count == 0)
			break;

		flags_different_than_last(keys.get(), helper.get(), current_count);
		inclusive_scan(helper, helper + current_count, helper);
		segment_size = compute_segment_size(helper.get(), current_count);
	}

	device_free(keys);
	device_free(destinations);
	device_free(helper);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "All sorting took " << milliseconds << " milliseconds" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return output.get();
}
