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

using namespace thrust;
using namespace std;

__global__ void create_hashes(uchar* words, int* word_positions, uint* segments, ullong* keys, const int offset,
	const int chars_to_hash, const int seg_shift, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const uint segment = segments[thread_num];
	auto key = static_cast<ullong>(segment) << seg_shift;
	const int my_position = word_positions[thread_num] + offset;

	if (my_position == -1)
		return;

	int i = 0;
	ullong hash = 0;
	for (; i < chars_to_hash; i++)
	{
		const unsigned char c = words[i + my_position];
		if (c == BREAKCHAR)
		{
			break;
		}
		hash *= ALPHABETSIZE;
		if (c >= ASCIILOWSTART)
			hash += c - ASCIILOWSTART;
		else
			hash += c - ASCIIUPSTART;
	}

	const ullong mask = words[i + my_position] == BREAKCHAR ? 0 : 1;

	for (; i < chars_to_hash; i++)
		hash *= ALPHABETSIZE;

	hash <<= 1;
	hash |= mask;

	key |= hash;
	keys[thread_num] = key;
}

__global__ void mark_singletons(ullong* keys, uchar* flags, int* destinations,
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

__global__ void create_consecutive_numbers(int* numbers, const int maxNumber)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= maxNumber)
		return;

	numbers[thread_num] = thread_num;
}

__global__ void ScatterValues(ullong* keysIn, ullong* keysOut, int* wordPositionsIn, int* wordPositionsOut,
	int* destinationsIn, int* destinationsOut, uchar* flags, int* positions, int wordCount)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= wordCount)
		return;

	if (!flags[thread_num])
		return;

	const int position = positions[thread_num];
	destinationsOut[position] = destinationsIn[thread_num];
	wordPositionsOut[position] = wordPositionsIn[thread_num];
	keysOut[position] = keysIn[thread_num];
}

__global__ void ComputeSegments(ullong* keys, uint* segments, int wordCount)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= wordCount)
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

	uint* d_segments;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_segments), sizeof(uint)*word_count));
	checkCudaErrors(cudaMemset(d_segments, 0, sizeof(int)*word_count));

	create_hashes << <num_blocks, num_threads >> > (d_word_array, d_sorted_positions, d_segments, d_hash_array_all, 0, CHARSTOHASH, KEYBITS, word_count);

	ComputeSegments << <num_blocks, num_threads >> > (d_hash_array_all, reinterpret_cast<uint*>(d_lengths), word_count);

	exclusive_scan(device_ptr<int>(d_lengths), device_ptr<int>(d_lengths + word_count), device_ptr<int>(d_sorted_positions));

	int output_hashes_count;
	checkCudaErrors(cudaMemcpy(&output_hashes_count, d_sorted_positions + word_count - 1, 4, cudaMemcpyDeviceToHost));
	output_hashes_count += 1;
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

int* get_sorted_positions(unsigned char* h_wordArray, int* h_wordPositions, int* h_wordLengths, const int word_count,
	size_t wordArraySize)
{
	ullong* d_keysIn;
	ullong* d_keysOut;
	unsigned char* d_wordArray;
	int* d_wordPositionsIn;
	int* d_wordPositionsOut;
	int* d_destinationsIn;
	int* d_destinationsOut;
	uint* d_segments;
	uchar* d_flags;
	int* d_scatterMap;
	int* d_output;
	checkCudaErrors(cudaMalloc((void**)&d_keysIn, sizeof(ullong)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_keysOut, sizeof(ullong)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_wordArray, wordArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositionsIn, sizeof(int)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositionsOut, sizeof(int)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_destinationsIn, sizeof(int)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_destinationsOut, sizeof(int)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_segments, sizeof(uint)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_flags, sizeof(uchar)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_scatterMap, sizeof(int)*word_count));
	checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(int)*word_count));

	checkCudaErrors(cudaMemcpy(d_wordArray, h_wordArray, wordArraySize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wordPositionsIn, h_wordPositions, word_count * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(d_segments, 0, sizeof(int)*word_count));
	checkCudaErrors(cudaMemset(d_flags, 0, sizeof(uchar)*word_count));

	uint numThreads, numBlocks;
	compute_grid_size(word_count, BLOCKSIZE, numBlocks, numThreads);
	const int max_length = *max_element(host, h_wordLengths, h_wordLengths + word_count);

	create_consecutive_numbers << <numBlocks, numThreads >> >(d_destinationsIn, word_count);
	int offset = 0;
	int segmentSize = 0;

	int currentCount = word_count;

	while (offset < max_length)
	{
		const int seg_chars = ceil(static_cast<double>(segmentSize) / CHARBITS);
		create_hashes << <numBlocks, numThreads >> > (d_wordArray, d_wordPositionsIn, d_segments, d_keysIn, offset, CHARSTOHASH - seg_chars, KEYBITS - segmentSize, currentCount);
		offset += CHARSTOHASH - seg_chars;

		sort_by_key(device_ptr<ullong>(d_keysIn), device_ptr<ullong>(d_keysIn + currentCount),
			device_ptr<int>(d_wordPositionsIn));
		
		mark_singletons << <numBlocks, numThreads >> > (d_keysIn, d_flags, d_destinationsIn,
			d_output, d_wordPositionsIn, currentCount);

		exclusive_scan(device_ptr<uchar>(d_flags),
			device_ptr<uchar>(d_flags + currentCount), device_ptr<int>(d_scatterMap));

		ScatterValues << <numBlocks, numThreads >> > (d_keysIn, d_keysOut, d_wordPositionsIn, d_wordPositionsOut,
			d_destinationsIn, d_destinationsOut, d_flags, d_scatterMap, currentCount);

		uchar last_flag;
		checkCudaErrors(cudaMemcpy(&last_flag, d_flags + currentCount - 1, 1, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&currentCount, d_scatterMap + currentCount - 1, 4, cudaMemcpyDeviceToHost));
		if (last_flag)
			currentCount += 1;
		if (currentCount == 0)
			break;

		compute_grid_size(currentCount, BLOCKSIZE, numBlocks, numThreads);

		ComputeSegments << <numBlocks, numThreads >> > (d_keysOut, d_segments, currentCount);

		inclusive_scan(device_ptr<uint>(d_segments),
			device_ptr<uint>(d_segments + currentCount), device_ptr<uint>(d_segments));

		uint max_segment;
		checkCudaErrors(cudaMemcpy(&max_segment, d_segments + currentCount - 1, sizeof(int), cudaMemcpyDeviceToHost));
		segmentSize = get_segment_size(max_segment);

		void* tmp = d_keysOut;
		d_keysOut = d_keysIn;
		d_keysIn = static_cast<ullong*>(tmp);
		tmp = d_wordPositionsOut;
		d_wordPositionsOut = d_wordPositionsIn;
		d_wordPositionsIn = static_cast<int*>(tmp);
		tmp = d_destinationsOut;
		d_destinationsOut = d_destinationsIn;
		d_destinationsIn = static_cast<int*>(tmp);
	}

	return d_output;
}
