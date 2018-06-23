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

#define CHARSTOHASH 6
#define ALPHABETSIZE 27
#define ASCIILOWSTART 96
#define ASCIIUPSTART 64
#define BREAKCHAR ' '
#define BLOCKSIZE 1
#define KEYLENGTH 8

typedef unsigned char uchar;
typedef unsigned long long ullong;

ullong ComputeHash(uchar * d_wordArray, int myPosition, int charsToHash, int offset)
{
	int i = 0;
	ullong hash = 0;
	for (; i < charsToHash; i++)
	{
		unsigned char c = d_wordArray[i + myPosition];
		if (c == BREAKCHAR)
			break;
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

	int wordLength = d_wordLengths[threadNum];
	int myPosition = d_wordPositions[threadNum];
	ullong hash = ComputeHash(d_wordArray, myPosition, CHARSTOHASH, 0);

	d_hashesOut[threadNum] = hash;
	indicesOut[threadNum] = threadNum;
}

__global__ void PackKeysD(uchar* words, int* wordPositions, uint* segments,
	ullong* keys, int offset, int charsToHash, int segShift, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	ullong key = segments[threadNum];
	key = key << segShift;
	int myPosition = wordPositions[threadNum];
	ullong hash = ComputeHash( words, myPosition, charsToHash, offset);
	key |= hash;
	keys[threadNum] = key;
}

__global__ void MarkSingletons(ullong* keys, uchar* flags, int* destinations, 
	int* output, int* wordStarts, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	uchar shouldOutput = 0;

	if (threadNum == 0)
	{
		shouldOutput = keys[threadNum] != keys[threadNum + 1];
	}
	else if (threadNum == wordCount - 1)
	{
		shouldOutput = keys[threadNum] != keys[threadNum - 1];
	}
	else
	{
		shouldOutput = keys[threadNum] != keys[threadNum - 1] && keys[threadNum] != keys[threadNum + 1];
	}

	flags[threadNum] = !shouldOutput;
	if (!shouldOutput)
		return;

	int indexOutput = destinations[threadNum];
	int wordStart = wordStarts[threadNum];
	output[indexOutput] = wordStart;
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
	int* d_destinationsOut;
	uint* d_segments;
	uchar* d_flags;
	int* d_wordPositionsOut;
	int* d_scatterMap;
	checkCudaErrors(cudaMalloc((void**)&d_keysIn, sizeof(ullong)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_keysOut, sizeof(ullong)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_wordArray, wordArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositionsIn, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositionsOut, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_destinationsIn, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_destinationsOut, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_segments, sizeof(uint)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_flags, sizeof(uchar)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositionsOut, sizeof(int)*wordCount));

	checkCudaErrors(cudaMemcpy(d_wordArray, h_wordArray, wordArraySize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wordPositionsIn, h_wordPositions, wordCount * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(d_segments, 0, sizeof(int)*wordCount));
	checkCudaErrors(cudaMemset(d_flags, 0, sizeof(uchar)*wordCount));

	uint numThreads, numBlocks;
	computeGridSize(wordCount, BLOCKSIZE, numBlocks, numThreads);

	CreateConsecutiveNumbers(d_destinationsIn, wordCount);
	int offset = 13;
	int segmentSize = 0;

	PackKeysD << <numBlocks, numThreads >> > (d_wordArray, d_wordPositions, d_segments, d_keys, 0, offset, 0, wordCount);
	int currentCount = wordCount;
	int maxLen = *(thrust::max_element(thrust::host, h_wordLengths, h_wordLengths + wordCount));
	while (offset < maxLen)
	{
		thrust::sort_by_key(thrust::device_ptr<uint>(d_keysIn), thrust::device_ptr<uint>(d_keysIn + currentCount),
			thrust::device_ptr<int>(d_wordPositionsIn));

		MarkSingletons << <numBlocks, numThreads >> > (d_keysIn, d_flags, d_destinationsIn,
			d_wordPositionsOut, d_wordPositionsIn, currentCount);
		thrust::exclusive_scan(thrust::device_ptr<uchar>(d_flags),
			thrust::device_ptr<uchar>(d_flags + currentCount), thrust::device_ptr<int>(d_scatterMap));
		ScatterValues << <numBlocks, numThreads >> > (d_keysIn, d_keysOut, d_wordPositionsIn, d_destinationsOut,
			d_destinationsIn, d_destinationsOut, d_flags, d_scatterMap, currentCount);
		checkCudaErrors(cudaMemcpy(&currentCount, d_scatterMap + sizeof(int)*(currentCount - 1), sizeof(int),
			cudaMemcpyDeviceToHost));
		currentCount += 1;
		ComputeSegments(d_keysOut, d_segments, currentCount);
		thrust::exclusive_scan(thrust::device_ptr<uint>(d_segments),
			thrust::device_ptr<uint>(d_segments + currentCount), thrust::device_ptr<uint>(d_segments));
		checkCudaErrors(cudaMemcpy(&segmentSize, d_segments + sizeof(int)*(currentCount - 1), sizeof(int),
			cudaMemcpyDeviceToHost));
		PackKeysD << <numBlocks, numThreads >> > (d_wordArray, d_wordPositions, d_segments, d_keys, offset, 13 - segmentSize, 64 - segmentSize, wordCount);
		offset += 13 - segmentSize;
	}
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

void SortStrings(unsigned char* h_wordArray, int* h_wordPositions, int* h_wordLengths, int wordCount, size_t wordArraySize)
{
	unsigned char* d_wordArrayIn;
	unsigned char* d_wordArrayOut;
	int* d_wordPositions;
	uint* d_hashes;
	int* d_wordLengths;
	int* d_wordPositionsResult;
	int* d_wordLengthsResult;
	int* d_indices;
	checkCudaErrors(cudaMalloc((void**)&d_wordArrayIn, wordArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_wordArrayOut, wordArraySize));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositions, sizeof(int)*wordCount));
	checkCudaErrors(cudaMalloc((void**)&d_hashes, wordCount * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&d_wordLengths, wordCount * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_wordPositionsResult, wordCount * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_wordLengthsResult, wordCount * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_indices, wordCount * sizeof(int)));

	checkCudaErrors(cudaMemcpy(d_wordLengths, h_wordLengths, wordCount * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wordPositions, h_wordPositions, wordCount * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wordArrayIn, h_wordArray, wordArraySize, cudaMemcpyHostToDevice));

	uint numThreads, numBlocks;
	computeGridSize(wordCount, BLOCKSIZE, numBlocks, numThreads);

	ComputeHashesD << <numBlocks, numThreads >> > (d_wordArrayIn, d_wordPositions, d_wordLengths, d_hashes, d_indices, wordCount);
	getLastCudaError("ComputeHashesD kernel execution failed.\n");

	thrust::sort_by_key(thrust::device_ptr<uint>(d_hashes), thrust::device_ptr<uint>(d_hashes + wordCount), thrust::device_ptr<int>(d_indices));

	RepositionDataD << <numBlocks, numThreads >> > (d_indices, d_wordPositionsResult, d_wordPositions, d_wordLengths, d_wordLengthsResult, wordCount);
	getLastCudaError("RepositionPositionsAndLengthsD kernel execution failed.\n");

	thrust::exclusive_scan(thrust::device_ptr<int>(d_wordLengthsResult), thrust::device_ptr<int>(d_wordLengthsResult + wordCount), thrust::device_ptr<int>(d_wordPositions));

	RepositionStringsD << <numBlocks, numThreads >> > (d_wordArrayIn, d_wordArrayOut, d_wordPositionsResult, d_wordPositions, d_wordLengthsResult, wordCount);
	getLastCudaError("RepositionDataD kernel execution failed.\n");

	checkCudaErrors(cudaMemcpy(h_wordArray, d_wordArrayOut, wordArraySize, cudaMemcpyDeviceToHost));
}