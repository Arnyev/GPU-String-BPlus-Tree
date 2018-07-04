#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include "helper_math.h"
#include "thrust/sort.h"
#include <thrust/execution_policy.h>
#include "thrust/device_ptr.h"
#include <thrust/extrema.h>
#include "parameters.h"
#include "functions.h"
#include <thrust/adjacent_difference.h>

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

__global__  void create_hashes_simple(uchar* words, int* word_positions, ullong* keys, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const int my_position = word_positions[thread_num];
	keys[thread_num] = get_hash(words, CHARSTOHASH, my_position);
}

__global__  void create_hashes_with_segD(uchar* words, int* word_positions, int* segments, ullong* keys, const int offset,
	const int chars_to_hash, const int seg_shift, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const int my_position = word_positions[thread_num] + offset;

	if (my_position == -1)
		return;

	keys[thread_num] = static_cast<ullong>(segments[thread_num]) << seg_shift | get_hash(words, chars_to_hash, my_position);
}

template <class T1>
__global__ void mark_singletonsD(ullong* keys, T1* flags, int* destinations,
	int* output, int* word_starts, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const ullong key = keys[thread_num];
	const int word_start = word_starts[thread_num];
	const bool finished = (key & 1ULL) == 0ULL;
	const int index_output = destinations[thread_num];

	if (thread_num == 0)
	{
		if (finished || key != keys[thread_num + 1])
		{
			output[index_output] = word_start;
			flags[thread_num] = 0;
		}
		else
			flags[thread_num] = 1;

		return;
	}

	const ullong key_last = keys[thread_num - 1];

	if (thread_num == word_count - 1)
	{
		if (key != key_last)
		{
			output[index_output] = word_start;
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
		output[index_output] = word_start;
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

__global__ void create_consecutive_numbersD(int* numbers, const int maxNumber)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= maxNumber)
		return;

	numbers[thread_num] = thread_num;
}

__global__ void scatter_values_d(ullong* keys_in, ullong* keys_out, int* word_positions_in, int* word_positions_out,
	int* destinations_in, int* destinations_out, uchar* flags, int* positions, const int wordCount)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= wordCount)
		return;

	if (!flags[thread_num])
		return;

	const int position = positions[thread_num];
	destinations_out[position] = destinations_in[thread_num];
	word_positions_out[position] = word_positions_in[thread_num];
	keys_out[position] = keys_in[thread_num];
}

template <class T1,class T2>
__global__ void flag_different_than_lastD(T1* keys, T2* segments, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	segments[thread_num] = 0;

	if (keys[thread_num] == 0)
	{
		return;
	}

	if (thread_num == 0)
	{
		segments[thread_num] = 1;
		return;
	}

	if (keys[thread_num] != keys[thread_num - 1])
		segments[thread_num] = 1;
}

__global__ void compute_lengths(uchar* words, int* positions, const int word_count, int* lengths)
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

__global__ void copy_suffixes(uchar* words, int* positions, const int word_count, int* lengths, uchar*suffixes, int * suffix_positions)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const int len = lengths[thread_num];
	if (len == 0)
		return;

	const int suffix_pos = suffix_positions[thread_num];
	const int position = positions[thread_num] + CHARSTOHASH;

	for (int i = 0; i < len; i++)
		suffixes[suffix_pos + i] = words[position + i];
}

__global__ void copy_values(int* input_suffix_positions, int* destinations, int* output_suffix_positions, int* flags, ullong* input_hashes, ullong* output_hashes, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	if (!flags[thread_num])
		return;

	const int output_position = destinations[thread_num];
	output_suffix_positions[output_position] = input_suffix_positions[thread_num];
	output_hashes[output_position] = input_hashes[thread_num];
}

__global__ void check_value(int* positions, uchar* flags, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	if (positions[thread_num] == -1)
		flags[thread_num] = 0;
	else
		flags[thread_num] = 1;
}

__global__ void move_positions(int* input, int* output_positions, int* output, uchar* flags, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	if (!flags[thread_num])
		return;

	output[output_positions[thread_num]] = input[thread_num];
}

sorting_output create_output(unsigned char* d_word_array, int* d_sorted_positions, int word_count)
{
	int* d_lengths;
	ullong* d_hash_array_all;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_hash_array_all), sizeof(ullong)*word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_lengths), sizeof(int)*word_count));

	int* d_suffix_positions;
	int* d_new_positions;
	uchar* d_flags;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_new_positions), sizeof(int)*word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_flags), sizeof(uchar)*word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_suffix_positions), sizeof(int)*word_count));

	uint num_threads, num_blocks;
	compute_grid_size(word_count, BLOCKSIZE, num_blocks, num_threads);

	check_value << <num_blocks, num_threads >> > (d_sorted_positions, d_flags, word_count);
	uchar lastflag;
	checkCudaErrors(cudaMemcpy(&lastflag, d_flags + word_count - 1, sizeof(uchar), cudaMemcpyDeviceToHost));

	exclusive_scan(device_ptr<uchar>(d_flags), device_ptr<uchar>(d_flags + word_count), device_ptr<int>(d_lengths));

	int new_word_count;
	checkCudaErrors(cudaMemcpy(&new_word_count, d_lengths + word_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
	new_word_count += lastflag;
	move_positions << <num_blocks, num_threads >> > (d_sorted_positions, d_lengths, d_new_positions, d_flags, word_count);

	checkCudaErrors(cudaMemset(d_lengths, 0, sizeof(int)*word_count));
	d_sorted_positions = d_new_positions;
	word_count = new_word_count;
	compute_grid_size(word_count, BLOCKSIZE, num_blocks, num_threads);

	compute_lengths << <num_blocks, num_threads >> > (d_word_array, d_sorted_positions, word_count, d_lengths);

	int last_suffix_length;
	checkCudaErrors(cudaMemcpy(&last_suffix_length, d_lengths + word_count - 1, sizeof(int), cudaMemcpyDeviceToHost));

	exclusive_scan(device_ptr<int>(d_lengths),
		device_ptr<int>(d_lengths + word_count), device_ptr<int>(d_suffix_positions));

	int output_size;
	checkCudaErrors(cudaMemcpy(&output_size, d_suffix_positions + word_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
	output_size += last_suffix_length;

	uchar* d_suffixes;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_suffixes), sizeof(uchar)*output_size));
	copy_suffixes << <num_blocks, num_threads >> > (d_word_array, d_sorted_positions, word_count, d_lengths, d_suffixes, d_suffix_positions);

	create_hashes_simple << <num_blocks, num_threads >> > (d_word_array, d_sorted_positions, d_hash_array_all, word_count);

	flag_different_than_lastD << <num_blocks, num_threads >> > (d_hash_array_all, reinterpret_cast<uint*>(d_lengths), word_count);

	int last_hash_different;
	checkCudaErrors(cudaMemcpy(&last_hash_different, d_lengths + word_count - 1, 4, cudaMemcpyDeviceToHost));

	exclusive_scan(device_ptr<int>(d_lengths), device_ptr<int>(d_lengths + word_count), device_ptr<int>(d_sorted_positions));

	int output_hashes_count;
	checkCudaErrors(cudaMemcpy(&output_hashes_count, d_sorted_positions + word_count - 1, 4, cudaMemcpyDeviceToHost));
	output_hashes_count += last_hash_different;
	int *d_output_positions;
	ullong* d_hashes;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_output_positions), sizeof(int)*output_hashes_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_hashes), sizeof(ullong)*output_hashes_count));
	copy_values << <num_blocks, num_threads >> > (d_suffix_positions, d_sorted_positions, d_output_positions, d_lengths, d_hash_array_all, d_hashes, word_count);

	return { d_hashes,d_output_positions,d_suffixes,output_hashes_count,output_size };
}

__global__ void RepositionStringsD(unsigned char* d_wordArrayIn, unsigned char* d_wordArrayOut, int* d_positionIn, int* d_positionOut, int wordCount)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= wordCount)
		return;

	const int positionIn = d_positionIn[thread_num];
	const int positionOut = d_positionOut[thread_num];

	int i = 0;
	char c;
	do
	{
		c = d_wordArrayIn[positionIn + i];
		d_wordArrayOut[positionOut + i] = c;
		i++;
	} while (c != BREAKCHAR);
}

struct different_than_last_binary : thrust::binary_function<ullong, ullong, ullong>
{
	__device__ ullong operator()(ullong v1, ullong v2) const
	{
		return v1 == v2 ? 0ULL : 1ULL;
	}
};

void scatter_values(ullong* &d_keys, int*& d_destinations, int* &d_word_positions, int old_word_count, int new_word_count, uchar* d_flags, int * d_scatter_map)
{
	uint num_threads, num_blocks;
	compute_grid_size(old_word_count, BLOCKSIZE, num_blocks, num_threads);

	ullong* d_keys_new;
	int* d_destinations_new;
	int* d_word_positions_new;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_keys_new), sizeof(ullong)*new_word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_word_positions_new), sizeof(int)*new_word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_destinations_new), sizeof(int)*new_word_count));

	scatter_values_d << <num_blocks, num_threads >> > (d_keys, d_keys_new, d_word_positions, d_word_positions_new,
		d_destinations, d_destinations_new, d_flags, d_scatter_map, old_word_count);

	getLastCudaError("Scatter values failed.");
	cudaDeviceSynchronize();
	checkCudaErrors(cudaFree(d_destinations));
	checkCudaErrors(cudaFree(d_keys));
	checkCudaErrors(cudaFree(d_word_positions));

	d_destinations = d_destinations_new;
	d_word_positions = d_word_positions_new;
	d_keys = d_keys_new;
}

int get_new_current_count(uchar* d_flags, int* d_helper, const int old_count)
{
	//cudaEvent_t start;
	//cudaEvent_t stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//float milliseconds = 0;
	//cudaDeviceSynchronize();
	//cudaEventRecord(start);

	int current_count;
	uchar last_flag;
	checkCudaErrors(cudaMemcpy(&last_flag, d_flags + old_count - 1, 1, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(&current_count, d_helper + old_count - 1, 4, cudaMemcpyDeviceToHost));
	cout << "Current count is " << current_count << endl;
	if (last_flag)
		current_count += 1;

	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//cout << "Getting count took " << milliseconds << " milliseconds" << endl;
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

	return current_count;
}

template <class T1>
void mark_singletons(ullong* d_keys, int* d_word_positions, int* d_destinations, T1* d_flags, int* d_output, int current_count)
{
	//cudaEvent_t start;
	//cudaEvent_t stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);
	//float milliseconds;

	uint num_threads;
	uint num_blocks;
	compute_grid_size(current_count, BLOCKSIZE, num_blocks, num_threads);
	mark_singletonsD << <num_blocks, num_threads >> > (d_keys, d_flags, d_destinations, d_output, d_word_positions, current_count);

	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//cout << "Singletons took " << milliseconds << " milliseconds" << endl;
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
}

template <class T1>
void flags_different_than_last(ullong* d_keys, T1* d_flags, int current_count)
{
	//cudaEvent_t start;
	//cudaEvent_t stop;
	//float milliseconds;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);

	uint num_threads;
	uint num_blocks;
	compute_grid_size(current_count, BLOCKSIZE, num_blocks, num_threads);
	flag_different_than_lastD << <num_blocks, num_threads >> > (d_keys, d_flags, current_count);
	//transform(device, device_ptr<ullong>(d_keysOut + 1), device_ptr<ullong>(d_keysOut + currentCount),
	//	device_ptr<ullong>(d_keysOut), device_ptr<ullong>(d_segments + 1), different_than_last_binary());

	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//cout << "Finding different than last took " << milliseconds << " milliseconds" << endl;
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
}

int compute_segment_size(int* d_helper, const int current_count)
{
	int max_segment;
	checkCudaErrors(cudaMemcpy(&max_segment, d_helper + current_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
	return get_segment_size(max_segment);
}

void create_hashes_with_seg(ullong* d_keys, unsigned char* d_word_array, int* d_word_positions, int* d_helper, int offset, int segment_size, int current_count, const int seg_chars)
{
	//cudaEvent_t start;
	//cudaEvent_t stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);
	//float milliseconds;

	uint num_threads;
	uint num_blocks;
	compute_grid_size(current_count, BLOCKSIZE, num_blocks, num_threads);
	create_hashes_with_segD << <num_blocks, num_threads >> > (d_word_array, d_word_positions, d_helper, d_keys, offset,
	                                                          CHARSTOHASH - seg_chars, KEYBITS - segment_size, current_count);
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//cout << "Create hashes took " << milliseconds << " milliseconds" << endl;
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
}

void create_consecutive_numbers(const int word_count, int* d_destinations)
{
	uint num_threads;
	uint num_blocks;
	compute_grid_size(word_count, BLOCKSIZE, num_blocks, num_threads);
	create_consecutive_numbersD << <num_blocks, num_threads >> > (d_destinations, word_count);
}

int* get_sorted_positions(int* d_word_positions, const int word_count, unsigned char* d_word_array)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;
	cudaEventRecord(start);

	ullong* d_keys;
	int* d_destinations;
	uchar* d_flags;
	int* d_helper;
	int* d_output;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_keys), sizeof(ullong)*word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_destinations), sizeof(int)*word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_flags), sizeof(uchar)*word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_helper), sizeof(int)*word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(int)*word_count));

	create_consecutive_numbers(word_count, d_destinations);

	int offset = 0;
	int segment_size = 0;
	int current_count = word_count;

	while (true)
	{
		const int seg_chars = ceil(static_cast<double>(segment_size) / CHARBITS);

		create_hashes_with_seg(d_keys, d_word_array, d_word_positions, d_helper, offset, segment_size, current_count, seg_chars);

		offset += CHARSTOHASH - seg_chars;

		sort_by_key(device_ptr<ullong>(d_keys), device_ptr<ullong>(d_keys + current_count), device_ptr<int>(d_word_positions));

		mark_singletons(d_keys, d_word_positions, d_destinations, d_flags, d_output, current_count);

		exclusive_scan(device_ptr<uchar>(d_flags), device_ptr<uchar>(d_flags + current_count), device_ptr<int>(d_helper));

		const int new_count = get_new_current_count(d_flags, d_helper, current_count);
		scatter_values(d_keys, d_destinations, d_word_positions, current_count, new_count, d_flags, d_helper);

		current_count = new_count;
		if (current_count == 0)
			break;

		flags_different_than_last(d_keys, d_helper, current_count);

		inclusive_scan(device_ptr<int>(d_helper), device_ptr<int>(d_helper + current_count), device_ptr<int>(d_helper));
		segment_size = compute_segment_size(d_helper, current_count);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "All sorting took " << milliseconds << " milliseconds" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return d_output;
}
