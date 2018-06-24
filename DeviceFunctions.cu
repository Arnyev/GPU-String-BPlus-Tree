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
#include <vector>

#define CHARSTOHASH 6
#define ALPHABETSIZE 27
#define ASCIILOWSTART 96
#define ASCIIUPSTART 64
#define BREAKCHAR ' '
#define BLOCKSIZE 1
#define KEYLENGTH 8

typedef unsigned char uchar;
typedef unsigned long long ullong;

__device__ __host__ ullong ComputeHash(uchar * d_wordArray, int myPosition, int charsToHash, int offset, bool* wordFinished)
{
	int i = 0;
	ullong hash = 0;
	for (; i < charsToHash; i++)
	{
		unsigned char c = d_wordArray[i + myPosition];
		if (c == BREAKCHAR) 
		{
			*wordFinished = true;
			break;
		}
		hash *= ALPHABETSIZE;
		if (c >= ASCIILOWSTART)
			hash += c - ASCIILOWSTART;
		else
			hash += c - ASCIIUPSTART;
	}

	for (; i < charsToHash; i++)
		hash *= ALPHABETSIZE;

	return hash;
}

__global__ void ComputeHashesD(unsigned char* d_wordArray, int* d_wordPositions, int* d_wordLengths, ullong* d_hashesOut, int* indicesOut, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	int myPosition = d_wordPositions[threadNum];
	bool wordFinished;
	ullong hash = ComputeHash(d_wordArray, myPosition, CHARSTOHASH, 0, &wordFinished);

	d_hashesOut[threadNum] = hash;
	indicesOut[threadNum] = threadNum;
}

__global__ void PackKeysD(uchar* words, int* wordPositions, uint* segments, bool * wordsFinished,
	ullong* keys, int offset, int charsToHash, int segShift, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	uint segment= segments[threadNum];
	ullong key = ((ullong)segment) << segShift;
	int myPosition = wordPositions[threadNum];
	bool wordFinished = false;
	ullong hash = ComputeHash(words, myPosition, charsToHash, offset, &wordFinished);
	wordsFinished[threadNum] = wordFinished;
	key |= hash;
	keys[threadNum] = key;
}

__global__ void MarkSingletons(ullong* keys, uchar* flags, int* destinations,
	int* output, int* wordStarts, int wordCount, bool * wordsFinished)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	uchar shouldOutput = 0;
	uchar isDuplicate = 0;
	ullong keyLast;
	ullong key;
	ullong keyNext;
	if (threadNum == 0)
	{
		shouldOutput = keys[threadNum] != keys[threadNum + 1];
	}
	else
	{
		keyLast = keys[threadNum - 1];
		key = keys[threadNum];
		if (threadNum == wordCount - 1)
			shouldOutput = keyLast != key;
		else
		{
			keyNext = keys[threadNum + 1];;
			shouldOutput = keyLast != key && key != keyNext;
		}

		isDuplicate = key == keyLast && wordsFinished[threadNum] && wordsFinished[threadNum - 1];
	}

	flags[threadNum] = !shouldOutput && !isDuplicate;
	if (!shouldOutput && !isDuplicate)
		return;

	int indexOutput = destinations[threadNum];
	int wordStart = wordStarts[threadNum];

	if (isDuplicate)
	{
		output[indexOutput] = -1;
	}
	else
	{
		output[indexOutput] = wordStart;
	}
}


void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
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

void SortStrings2(unsigned char* h_wordArray, int* h_wordPositions, int* h_wordLengths, int wordCount,
	size_t wordArraySize)
{
	ullong* d_keysIn;
	ullong* d_keysOut;
	unsigned char* d_wordArray;
	int* d_wordPositionsIn;
	int* d_wordPositionsOut;
	int* d_destinationsIn;
	bool* d_wordsFinished;
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
	checkCudaErrors(cudaMalloc((void**)&d_wordsFinished, sizeof(bool)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_scatterMap, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(int)*wordCount));

	checkCudaErrors(cudaMemcpy(d_wordArray, h_wordArray, wordArraySize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wordPositionsIn, h_wordPositions, wordCount * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(d_segments, 0, sizeof(int)*wordCount));
	checkCudaErrors(cudaMemset(d_flags, 0, sizeof(uchar)*wordCount));

	uint numThreads, numBlocks;
	computeGridSize(wordCount, BLOCKSIZE, numBlocks, numThreads);

	CreateConsecutiveNumbers << <numBlocks, numThreads >> >(d_destinationsIn, wordCount);
	int offset = 0;
	int segmentSize = 0;

	PackKeysD << <numBlocks, numThreads >> > (d_wordArray, d_wordPositionsIn, d_segments, d_wordsFinished, d_keysIn, 0, offset, 0, wordCount);

	ullong* keys = (ullong*)malloc(wordCount * sizeof(ullong));
	checkCudaErrors(cudaMemcpy(keys, d_keysIn, wordCount * sizeof(ullong), cudaMemcpyDeviceToHost));
	std::vector<ullong> v2(keys, keys + wordCount);

	int currentCount = wordCount;
	int maxLen = *(thrust::max_element(thrust::host, h_wordLengths, h_wordLengths + wordCount));
	while (offset < maxLen)
	{
		if (offset == 0)
			offset = 13;

		thrust::sort_by_key(thrust::device_ptr<ullong>(d_keysIn), thrust::device_ptr<ullong>(d_keysIn + currentCount),
			thrust::device_ptr<int>(d_wordPositionsIn));

		MarkSingletons << <numBlocks, numThreads >> > (d_keysIn, d_flags, d_destinationsIn,
			d_output, d_wordPositionsIn, (int)currentCount, d_wordsFinished);

		thrust::exclusive_scan(thrust::device_ptr<uchar>(d_flags),
			thrust::device_ptr<uchar>(d_flags + currentCount), thrust::device_ptr<int>(d_scatterMap));

		ScatterValues << <numBlocks, numThreads >> > (d_keysIn, d_keysOut, d_wordPositionsIn, d_destinationsOut,
			d_destinationsIn, d_destinationsOut, d_flags, d_scatterMap, currentCount);

		checkCudaErrors(cudaMemcpy(h_wordPositions, d_scatterMap, currentCount * sizeof(int), cudaMemcpyDeviceToHost));
		std::vector<int> v(h_wordPositions, h_wordPositions + wordCount);

		checkCudaErrors(cudaMemcpy(&currentCount, d_scatterMap + currentCount - 1, 4, cudaMemcpyDeviceToHost));
		currentCount += 1;
		computeGridSize((int)currentCount, BLOCKSIZE, numBlocks, numThreads);

		ComputeSegments << <numBlocks, numThreads >> > (d_keysOut, d_segments,(int) currentCount);

		thrust::exclusive_scan(thrust::device_ptr<uint>(d_segments),
			thrust::device_ptr<uint>(d_segments + currentCount), thrust::device_ptr<uint>(d_segments));

		checkCudaErrors(cudaMemcpy(&segmentSize, d_segments + currentCount - 1, sizeof(int), cudaMemcpyDeviceToHost));

		PackKeysD << <numBlocks, numThreads >> > (d_wordArray, d_wordPositionsOut, d_segments,
			d_wordsFinished, d_keysOut, offset, 13 - segmentSize, 64 - segmentSize, wordCount);

		offset += 13 - segmentSize;
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

	uchar* wordArrayOut = (uchar*)malloc(wordArraySize);

	checkCudaErrors(cudaMemcpy(h_wordPositions, d_output, wordCount * sizeof(int), cudaMemcpyDeviceToHost));
	int wordArrayIndex = 0;
	int indexInWord = 0;

	for (int i = 0; i < wordCount; i++)
	{
		int position = h_wordPositions[i];
		if (position == -1)
			continue;
		indexInWord = 0;
		while (true)
		{
			uchar c = h_wordArray[position + indexInWord];
			if (c != BREAKCHAR)
			{
				wordArrayOut[wordArrayIndex++] = c;
				indexInWord++;
			}
			else
				break;
		}
		wordArrayOut[wordArrayIndex++] = BREAKCHAR;
	}
	std::vector<int> pos(h_wordPositions, h_wordPositions + wordCount);
	std::vector<char> pos2(wordArrayOut, wordArrayOut + wordArraySize);
	std::vector<int>::iterator it;
	it = std::unique(pos.begin(), pos.end());   // 10 20 30 20 10 ?  ?  ?  ?//                
	pos.resize(std::distance(pos.begin(), it)); // 10 20 30 20 10
	std::string s(wordArrayOut, wordArrayOut + wordArraySize);
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

__global__ void RepositionDataD(int* d_indices, int* d_positionOut, int* d_positionIn, int* d_wordLenghtsIn, int* d_wordLenghtsOut, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	int oldIndex = d_indices[threadNum];
	d_wordLenghtsOut[threadNum] = d_wordLenghtsIn[oldIndex];
	d_positionOut[threadNum] = d_positionIn[oldIndex];
}
