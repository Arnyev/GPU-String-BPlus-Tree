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

#define CHARSTOHASH 6
#define ALPHABETSIZE 26
#define ASCIILOWSTART 97
#define ASCIIUPSTART 65
#define BREAKCHAR ' '
#define BLOCKSIZE 1

__global__ void ComputeHashesD(unsigned char* d_wordArray, int* d_wordPositions, int* d_wordLengths, uint* d_hashesOut, int* indicesOut, int wordCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= wordCount)
		return;

	int wordLength = d_wordLengths[threadNum];
	int myPosition = d_wordPositions[threadNum];
	uint hash = 0;
	for (int i = 0; i < CHARSTOHASH && i < wordLength; i++)
	{
		unsigned char c = d_wordArray[i + myPosition];
		if (c == BREAKCHAR)
		{
			for (int j = 0; j < CHARSTOHASH - i; j++)
				hash *= ALPHABETSIZE;
		}
		else
		{
			hash *= ALPHABETSIZE;
			if (c >= ASCIILOWSTART)
				hash += c - ASCIILOWSTART;
			else
				hash += c - ASCIIUPSTART;
		}
	}

	d_hashesOut[threadNum] = hash;
	indicesOut[threadNum] = threadNum;
}

void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
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