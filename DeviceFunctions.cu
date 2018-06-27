#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "helper_math.h"
#include "thrust/sort.h"
#include <thrust/execution_policy.h>
#include "thrust/device_ptr.h"
#include "bplus_tree_gpu.cuh"
#include <thrust/extrema.h>
#include "SortingsTests.h"

#define CHARSTOHASH 12
#define ALPHABETSIZE 27
#define ASCIILOWSTART 96
#define ASCIIUPSTART 64
#define BLOCKSIZE 256
#define KEYBITS 64
#define CHARBITS 5

using namespace thrust;

__global__ void PackKeysD(uchar* words, int* wordPositions, uint* segments, 
	ullong* keys, int offset, int charsToHash, int segShift, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	uint segment= segments[threadNum];
	ullong key = ((ullong)segment) << segShift;
	int myPosition = wordPositions[threadNum] + offset;

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
	mask = words[i + myPosition] == BREAKCHAR ? 0 : 1;

	for (; i < charsToHash; i++)
		hash *= ALPHABETSIZE;

	hash <<= 1;
	hash |= mask;

	key |= hash;
	keys[threadNum] = key;
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

void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = blockSize < n ? blockSize : n;
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
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

	if (threadNum == 0 || keys[threadNum] != keys[threadNum - 1])
		segments[threadNum] = 1;
}

int get_segment_size(uint max_segment)
{
	int segment_size;
	if (max_segment == 0)
		segment_size = 0;
	else
	{
		segment_size = 32;
		const uint flag = 1UL << 31;
		while ((max_segment&flag) == 0)
		{
			max_segment <<= 1;
			segment_size--;
		}
	}

	return segment_size;
}

template <class T>
vector<T> create_vector(T* d_pointer, int size)
{
	T* ptr = static_cast<T*>(malloc(sizeof(T) * size));
	checkCudaErrors(cudaMemcpy(ptr, d_pointer, sizeof(T)*size, cudaMemcpyDeviceToHost));
	return vector<T>(ptr, ptr + size);
}

__global__ void compute_lengths(uchar* words, int* positions, const int word_count, int* lengths)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	int length = 0;
	uchar c;
	int my_position = positions[thread_num];
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
	const int position = positions[thread_num];

	for (int i = 0; i < len; i++)
		suffixes[suffix_pos + i] = words[position + i];
}


void CreateOutputArrays(unsigned char* d_wordArray, int* d_sortedPositions, const int word_count, ullong*& d_hash_array, uchar* & d_suffixes, int*&d_output_positions)
{
	int* d_lengths;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_hash_array), sizeof(ullong)*word_count));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_lengths), sizeof(int)*word_count));
	checkCudaErrors(cudaMemset(d_lengths, 0, sizeof(int)*word_count));

	int* d_suffix_positions;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_suffix_positions), sizeof(int)*word_count));

	uint num_threads, num_blocks;
	computeGridSize(word_count, BLOCKSIZE, num_blocks, num_threads);

	compute_lengths << <num_blocks, num_threads >> > (d_wordArray, d_sortedPositions, word_count, d_lengths);

	int last_len;
	checkCudaErrors(cudaMemcpy(&last_len, d_lengths + word_count - 1, sizeof(int), cudaMemcpyDeviceToHost));

	inclusive_scan(device_ptr<uint>(d_lengths),
		device_ptr<uint>(d_lengths + word_count), device_ptr<uint>(d_suffix_positions));

	int len_sum;

	checkCudaErrors(cudaMemcpy(&len_sum, d_suffix_positions + word_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
	const int output_size = last_len + len_sum;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_suffixes), sizeof(uchar)*output_size));
	copy_suffixes << <num_blocks, num_threads >> > (d_wordArray, d_sortedPositions, word_count, d_lengths, d_suffixes, d_suffix_positions);

	PackKeysD << <num_blocks, num_threads >> > (d_wordArray, d_sortedPositions, reinterpret_cast<uint*>(d_sortedPositions), d_hash_array, 0, CHARSTOHASH, KEYBITS, word_count);

}

void SortStrings(unsigned char* h_wordArray, int* h_wordPositions, int* h_wordLengths, int wordCount,
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
	computeGridSize(wordCount, BLOCKSIZE, numBlocks, numThreads);
	int maxLen = *(max_element(host, h_wordLengths, h_wordLengths + wordCount));

	CreateConsecutiveNumbers << <numBlocks, numThreads >> >(d_destinationsIn, wordCount);
	int offset = 0;
	int segmentSize = 0;

	int currentCount = wordCount;

	while (offset < maxLen)
	{
		int segChars = ceil(static_cast<double>(segmentSize) / CHARBITS);
		PackKeysD << <numBlocks, numThreads >> > (d_wordArray, d_wordPositionsIn, d_segments, d_keysIn, offset, CHARSTOHASH - segChars, KEYBITS - segmentSize, currentCount);
		offset += CHARSTOHASH - segChars;

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
		computeGridSize((int)currentCount, BLOCKSIZE, numBlocks, numThreads);

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

	test_string_sorting(h_wordPositions, d_output, wordCount, h_wordArray, wordArrayOut, wordArraySize);
}

__global__ void RepositionStringsD(unsigned char* d_wordArrayIn, unsigned char* d_wordArrayOut, int* d_positionIn, int* d_positionOut, int* d_wordLenghts, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	int positionIn = d_positionIn[threadNum];
	int positionOut = d_positionOut[threadNum];
	int wordLength = d_wordLenghts[threadNum];

	for (int i = 0; i < wordLength; i++)
		d_wordArrayOut[positionOut + i] = d_wordArrayIn[positionIn + i];
}
