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

__global__ void pack_keys_d(uchar* words, int* wordPositions, uint* segments, 
	ullong* keys, int offset, int charsToHash, int segShift, int wordCount)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= wordCount)
		return;

	//todo zepsute segmenty
	const uint segment= segments[thread_num];
	ullong key = static_cast<ullong>(segment) << segShift;
	const int myPosition = wordPositions[thread_num] + offset;

	if (myPosition == -1)
		return;

	ullong mask = 0;
	int i = 0;
	ullong hash = 0;
	for (; i < charsToHash; i++)
	{
		unsigned char c = words[i + myPosition];
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

	const unsigned char d = words[i + myPosition];
	if(d==BREAKCHAR)
	{
		mask = 0;
	}
	else
	{
		mask = 1;
	}
	//mask = words[i + myPosition] == BREAKCHAR ? 0 : 1;

	for (; i < charsToHash; i++)
		hash *= ALPHABETSIZE;

	hash <<= 1;
	hash |= mask;

	key |= hash;
	keys[thread_num] = key;
}

__global__ void MarkSingletons(ullong* keys, uchar* flags, int* destinations,
	int* output, int* wordStarts, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	ullong key = keys[threadNum];
	int wordStart = wordStarts[threadNum];
	const bool finished = (key & 1ULL) == 0ULL;
	int indexOutput = destinations[threadNum];

	if (threadNum == 0)
	{
		if (finished || key != keys[threadNum + 1])
		{
			output[indexOutput] = wordStart;
			flags[threadNum] = 0;
		}
		else
			flags[threadNum] = 1;

		return;
	}

	ullong keyLast = keys[threadNum - 1];

	if (threadNum == wordCount - 1)
	{
		if (key != keyLast)
		{
			output[indexOutput] = wordStart;
			flags[threadNum] = 0;
		}
		else if (finished)
		{
			output[indexOutput] = -1;
			flags[threadNum] = 0;
		}
		else
			flags[threadNum] = 1;

		return;
	}

	ullong keyNext = keys[threadNum + 1];

	if (key != keyLast && (finished || key != keyNext))
	{
		output[indexOutput] = wordStart;
		flags[threadNum] = 0;
	}
	else if (key == keyLast && finished)
	{
		output[indexOutput] = -1;
		flags[threadNum] = 0;
	}
	else
		flags[threadNum] = 1;
}

__global__ void CreateConsecutiveNumbers(int* numbers, int maxNumber)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= maxNumber)
		return;

	numbers[threadNum] = threadNum;
}

__global__ void ScatterValues(ullong* keysIn, ullong* keysOut, int* wordPositionsIn, int* wordPositionsOut,
	int* destinationsIn, int* destinationsOut, uchar* flags, int* positions, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	if (!flags[threadNum])
		return;

	int position = positions[threadNum];
	destinationsOut[position] = destinationsIn[threadNum];
	wordPositionsOut[position] = wordPositionsIn[threadNum];
	keysOut[position] = keysIn[threadNum];
}

__global__ void ComputeSegments(ullong* keys, uint* segments, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	segments[threadNum] = 0;

	if (keys[threadNum] == 0)
	{
		return;
	}

	if (threadNum == 0)
	{
		segments[threadNum] = 1;
		return;
	}

	if (keys[threadNum] != keys[threadNum - 1])
		segments[threadNum] = 1;
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

sorting_output create_output(unsigned char* d_wordArray, int char_count, int* d_sortedPositions, int word_count)
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

	check_value << <num_blocks, num_threads >> > (d_sortedPositions, d_flags, word_count);
	uchar lastflag;
	checkCudaErrors(cudaMemcpy(&lastflag, d_flags + word_count - 1, sizeof(uchar), cudaMemcpyDeviceToHost));

	exclusive_scan(device_ptr<uchar>(d_flags), device_ptr<uchar>(d_flags + word_count), device_ptr<int>(d_lengths));

	int new_word_count;
	checkCudaErrors(cudaMemcpy(&new_word_count, d_lengths + word_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
	new_word_count += lastflag;
	move_positions << <num_blocks, num_threads >> > (d_sortedPositions, d_lengths, d_new_positions, d_flags, word_count);

	checkCudaErrors(cudaMemset(d_lengths, 0, sizeof(int)*word_count));
	d_sortedPositions = d_new_positions;
	word_count = new_word_count;
	compute_grid_size(word_count, BLOCKSIZE, num_blocks, num_threads);

	compute_lengths << <num_blocks, num_threads >> > (d_wordArray, d_sortedPositions, word_count, d_lengths);

	int last_len;
	checkCudaErrors(cudaMemcpy(&last_len, d_lengths + word_count - 1, sizeof(int), cudaMemcpyDeviceToHost));

	exclusive_scan(device_ptr<int>(d_lengths),
		device_ptr<int>(d_lengths + word_count), device_ptr<int>(d_suffix_positions));
	auto vaf = create_vector(d_lengths, word_count);
	auto kaf = create_vector(d_suffix_positions, word_count);
	int len_sum;

	checkCudaErrors(cudaMemcpy(&len_sum, d_suffix_positions + word_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
	const int output_size = last_len + len_sum;

	uchar* d_suffixes;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_suffixes), sizeof(uchar)*output_size));
	copy_suffixes << <num_blocks, num_threads >> > (d_wordArray, d_sortedPositions, word_count, d_lengths, d_suffixes, d_suffix_positions);

	uint* d_segments;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_segments), sizeof(uint)*word_count));
	checkCudaErrors(cudaMemset(d_segments, 0, sizeof(int)*word_count));

	pack_keys_d << <num_blocks, num_threads >> > (d_wordArray, d_sortedPositions, d_segments, d_hash_array_all, 0, CHARSTOHASH, KEYBITS, word_count);

	cudaDeviceSynchronize();
	ComputeSegments << <num_blocks, num_threads >> > (d_hash_array_all, reinterpret_cast<uint*>(d_lengths), word_count);
	cudaDeviceSynchronize();

	exclusive_scan(device_ptr<int>(d_lengths), device_ptr<int>(d_lengths + word_count), device_ptr<int>(d_sortedPositions));

	int output_hashes_count;
	checkCudaErrors(cudaMemcpy(&output_hashes_count, d_sortedPositions + word_count - 1, 4, cudaMemcpyDeviceToHost));
	output_hashes_count += 1;
	int *d_output_positions;
	ullong* d_hashes;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_output_positions), sizeof(int)*output_hashes_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_hashes), sizeof(ullong)*output_hashes_count));
	copy_values << <num_blocks, num_threads >> > (d_suffix_positions, d_sortedPositions, d_output_positions, d_lengths, d_hash_array_all, d_hashes, word_count);

	return { d_hashes,d_output_positions,d_suffixes,output_hashes_count,output_size };
}

__global__ void RepositionStringsD(unsigned char* d_wordArrayIn, unsigned char* d_wordArrayOut, int* d_positionIn, int* d_positionOut, int wordCount)
{
	const int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	const int positionIn = d_positionIn[threadNum];
	const int positionOut = d_positionOut[threadNum];

	int i = 0;
	char c = BREAKCHAR;
	do
	{
		c = d_wordArrayIn[positionIn + i];
		d_wordArrayOut[positionOut + i] = c;
		i++;
	} while (c != BREAKCHAR);
}

int* get_sorted_positions(unsigned char* h_wordArray, int* h_wordPositions, int* h_wordLengths, int wordCount,
	size_t wordArraySize)
{
	sort_by_key(h_wordLengths, h_wordLengths + wordCount, h_wordPositions);

	uchar* wordArrayOut = (uchar*)malloc(wordArraySize);

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
	checkCudaErrors(cudaMalloc((void**)&d_keysIn, sizeof(ullong)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_keysOut, sizeof(ullong)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_wordArray, wordArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositionsIn, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositionsOut, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_destinationsIn, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_destinationsOut, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_segments, sizeof(uint)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_flags, sizeof(uchar)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_scatterMap, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(int)*wordCount));

	checkCudaErrors(cudaMemcpy(d_wordArray, h_wordArray, wordArraySize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wordPositionsIn, h_wordPositions, wordCount * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(d_segments, 0, sizeof(int)*wordCount));
	checkCudaErrors(cudaMemset(d_flags, 0, sizeof(uchar)*wordCount));

	uint numThreads, numBlocks;
	compute_grid_size(wordCount, BLOCKSIZE, numBlocks, numThreads);
	int maxLen = *(max_element(host, h_wordLengths, h_wordLengths + wordCount));

	CreateConsecutiveNumbers << <numBlocks, numThreads >> >(d_destinationsIn, wordCount);
	int offset = 0;
	int segmentSize = 0;

	int currentCount = wordCount;

	while (offset < maxLen)
	{
		const int seg_chars = ceil(static_cast<double>(segmentSize) / CHARBITS);
		pack_keys_d << <numBlocks, numThreads >> > (d_wordArray, d_wordPositionsIn, d_segments, d_keysIn, offset, CHARSTOHASH - seg_chars, KEYBITS - segmentSize, currentCount);
		offset += CHARSTOHASH - seg_chars;

		auto xa = create_vector(d_keysIn, currentCount);
		auto x2 = create_vector(d_wordPositionsIn, currentCount);

		sort_by_key(device_ptr<ullong>(d_keysIn), device_ptr<ullong>(d_keysIn + currentCount),
			device_ptr<int>(d_wordPositionsIn));
		auto ga = create_vector(d_keysIn, currentCount);
		auto x22 = create_vector(d_wordPositionsIn, currentCount);
		
		MarkSingletons << <numBlocks, numThreads >> > (d_keysIn, d_flags, d_destinationsIn,
			d_output, d_wordPositionsIn, currentCount);
		auto va = create_vector(d_flags, currentCount);
		auto xaa = create_vector(d_output,wordCount);
		exclusive_scan(device_ptr<uchar>(d_flags),
			device_ptr<uchar>(d_flags + currentCount), device_ptr<int>(d_scatterMap));

		ScatterValues << <numBlocks, numThreads >> > (d_keysIn, d_keysOut, d_wordPositionsIn, d_wordPositionsOut,
			d_destinationsIn, d_destinationsOut, d_flags, d_scatterMap, currentCount);

		auto zaaa = create_vector(d_destinationsOut, currentCount);
		auto vatt = create_vector(d_wordPositionsOut, currentCount);
		auto katt = create_vector(d_keysOut, currentCount);
		uchar f;
		checkCudaErrors(cudaMemcpy(&f, d_flags + currentCount - 1, 1, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&currentCount, d_scatterMap + currentCount - 1, 4, cudaMemcpyDeviceToHost));
		if (f)
			currentCount += 1;
		if (currentCount == 0)
			break;
		compute_grid_size((int)currentCount, BLOCKSIZE, numBlocks, numThreads);

		ComputeSegments << <numBlocks, numThreads >> > (d_keysOut, d_segments,(int) currentCount);
		auto aga = create_vector(d_segments, currentCount);

		inclusive_scan(device_ptr<uint>(d_segments),
			device_ptr<uint>(d_segments + currentCount), device_ptr<uint>(d_segments));
		auto zgaga = create_vector(d_segments, currentCount);

		uint max_segment;
		checkCudaErrors(cudaMemcpy(&max_segment, d_segments + currentCount - 1, sizeof(int), cudaMemcpyDeviceToHost));
		segmentSize = get_segment_size(max_segment);

		void* tmp = d_keysOut;
		d_keysOut = d_keysIn;
		d_keysIn = (ullong*)tmp;
		tmp = d_wordPositionsOut;
		d_wordPositionsOut = d_wordPositionsIn;
		d_wordPositionsIn = (int*)tmp;
		tmp = d_destinationsOut;
		d_destinationsOut = d_destinationsIn;
		d_destinationsIn = (int*)tmp;
	}

	return d_output;
}
