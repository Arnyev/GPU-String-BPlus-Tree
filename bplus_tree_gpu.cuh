#pragma once
#include <vector>
#include <cassert>
#include <algorithm>

#include "gpu_helper.cuh"
#include "bplus_tree.h"
#include "not_implemented.h"
#include "parameters.h"
#include "sort_helpers.cuh"
#include <thrust/binary_search.h>
#include "bplus_tree_gpu.cuh"

struct output_create_leafs
{
	int usedNodes;
	int isOnlyRoot;
	int rootNodeIndex;
};

struct output_create_next_layer
{
	int lastUsedIndex;
	int isRoot;
};

#pragma region build_tree_kernels
template<class HASH, int B>
__global__ void kernel_create_next_layer(const int threadsNum, const int beginIndex, const int endIndex, int* indexArray, HASH* keysArray, int* sizeArray, HASH *minArray)
{
	only_gpu_assert();
	const int globalId = GetGlobalId();
	const int minIndexesPerNode = B / 2 + 1;
	const int maxIndexesPerNode = B + 1;
	const int minKeysPerNode = B / 2;
	const int maxKeysPerNode = B;
	const int createdNodes = endIndex - beginIndex; //How many nodes were last time created
	//Creation of new layer
	int toCreate = my_max(1, createdNodes / (B / 2 + 1)); //How many nodes will be created in this iteration
	if (toCreate <= 0)
		toCreate = 1;
	//In each node there will be at least B / 2 keys and B / 2 + 1 indexes to lower layer nodes
	int id = globalId;
	while (id < createdNodes)
	{
		int _nodeNumber = id / minIndexesPerNode;
		const bool addToPrevious = _nodeNumber == toCreate;
		_nodeNumber += addToPrevious ? -1 : 0;
		const int& nodeNumber = _nodeNumber;
		const int nodeIndex = endIndex + nodeNumber;
		const int indexInNode = id - nodeNumber * minIndexesPerNode;
		const int lowerNode = beginIndex + id;
		indexArray[nodeIndex * maxIndexesPerNode + indexInNode] = lowerNode;
		if (indexInNode != 0)
		{
			keysArray[nodeIndex * maxKeysPerNode + indexInNode - 1] = minArray[lowerNode];
		}
		id += threadsNum;
	}
	//Filling size of nodes
	id = globalId;
	while (id < toCreate)
	{
		const int &nodeNumber = id;
		const int nodeIndex = endIndex + nodeNumber;
		const bool isLast = nodeNumber == toCreate - 1;
		const int firstLowerNode = beginIndex + id * minIndexesPerNode;
		sizeArray[nodeIndex] = isLast ? createdNodes - (toCreate - 1) * minIndexesPerNode - 1 : minKeysPerNode;
		minArray[nodeIndex] = minArray[firstLowerNode];
		id += threadsNum;
	}
}

template<class HASH, int B>
__global__ void kernel_create_leafs(const int threadsNum, const int elementNum, HASH* hashesArray, int* valueArray, HASH* keysArray, int* sizeArray, int* indexesArray, HASH* minArray)
{
	only_gpu_assert();
	const int globalId = GetGlobalId();
	const int maxIndexesPerNode = B + 1;
	const int minKeysPerNode = B / 2;
	const int maxKeysPerNode = B;
	int bottomPages = my_max(1, elementNum * 2 / B); //How many pages will be created
	const int elementsOnLastPage = elementNum - (bottomPages - 1) * B / 2;
	if (elementsOnLastPage < B / 2 && bottomPages > 1) //If elements on last page are less then half size of page
		bottomPages -= 1;
	int id = globalId;
	while (id < elementNum)
	{
		int _nodeIndex = id / minKeysPerNode;
		const bool addToPrevious = _nodeIndex == bottomPages;
		_nodeIndex += addToPrevious ? -1 : 0;
		const int& nodeIndex = _nodeIndex;
		const int indexInNode = id - nodeIndex * minKeysPerNode;
		keysArray[nodeIndex * maxKeysPerNode + indexInNode] = hashesArray[id];
		indexesArray[nodeIndex * maxIndexesPerNode + indexInNode] = valueArray[id];
		id += threadsNum;
	}
	id = globalId;
	while (id < bottomPages)
	{
		const int &nodeIndex = id;
		const bool isLast = nodeIndex == bottomPages - 1;
		sizeArray[nodeIndex] = isLast ? elementNum - (bottomPages - 1) * minKeysPerNode : minKeysPerNode;
		minArray[nodeIndex] = hashesArray[nodeIndex * minKeysPerNode];
		indexesArray[nodeIndex * maxIndexesPerNode + maxIndexesPerNode - 1] = isLast ? -1 : nodeIndex + 1;
		id += threadsNum;
	}
}

template<class HASH, int B, class Output>
__global__ void kernel_get_value(const int threadsNum, const int elementNum, HASH* keysArray, int* sizeArray, int* indexesArray, HASH* toFind, int height, int rootIndex, Output* output)
{
	only_gpu_assert();
	const int globalId = GetGlobalId();
	const int maxIndexesPerNode = B + 1;
	const int maxKeysPerNode = B;
	int id = globalId;
	while (id < elementNum)
	{
		const HASH key = toFind[id];
		int currentHeight = 0;
		int node = rootIndex;
		//Inner nodes
		while (currentHeight < height)
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys = keys_begin;
			while (keys < keys_end && *keys <= key)
			{
				++keys;
			}
			node = indexesArray[node * maxIndexesPerNode + (keys - keys_begin)];
			currentHeight += 1;
		}
		//Leaf level
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys = keys_begin;
			while (keys < keys_end && *keys < key)
			{
				++keys;
			}
			if (keys < keys_end && *keys == key)
			{
				if (std::is_same<Output, bool>::value)
					output[id] = true;
				else
					output[id] = indexesArray[node * maxIndexesPerNode + (keys - keys_begin)];
			}
			else
			{
				if (std::is_same<Output, bool>::value)
					output[id] = false;
				else
					output[id] = static_cast<Output>(-1);
			}
		}
		id += threadsNum;
	}
}
#pragma endregion

#pragma region search_kernels

template <class HASH, int B>
__global__ void kernel_find_words_v1(const int threadsNum, HASH* keysArray, int* indexesArray, int* sizeArray,
	const int rootIndex, const int height, char* suffixes, int suffixesSize,
	const int elementsNum, char* words, int* beginIndexes, bool* output)
{
	const int globalId = threadIdx.x + gridDim.x * blockIdx.x;
	const int maxIndexesPerNode = B + 1;
	const int maxKeysPerNode = B;
	int id = globalId;
	while (id < elementsNum)
	{
		const int beginIdx = beginIndexes[id];
		const HASH key = get_hash<HASH>(words, beginIdx);
		int currentHeight = 0;
		int node = rootIndex;
		//Inner nodes
		while (currentHeight < height)
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys = keys_begin;
			while (keys < keys_end && *keys <= key)
			{
				++keys;
			}
			node = indexesArray[node * maxIndexesPerNode + (keys - keys_begin)];
			currentHeight += 1;
		}
		int suffixIdx, endSuffixIdx = -1;
		//Leaf level
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys = keys_begin;
			while (keys < keys_end && *keys < key)
			{
				++keys;
			}
			if (keys < keys_end && *keys == key)
			{
				const int indexInKeyArray = keys - keys_begin;
				suffixIdx = indexesArray[node * maxIndexesPerNode + indexInKeyArray];
				if (indexInKeyArray < size - 1) //Next element is in the same leaf
				{
					endSuffixIdx = indexesArray[node * maxIndexesPerNode + indexInKeyArray + 1];
				}
				else //Next element is in the next leaf
				{
					if (indexesArray[node * maxIndexesPerNode + maxIndexesPerNode - 1] != -1) //Next leaf exists
					{
						endSuffixIdx = indexesArray[(node + 1) * maxIndexesPerNode];
					}
					else //It is the last element in the last leaf
					{
						endSuffixIdx = suffixesSize;
					}
				}
			}
			else
			{
				suffixIdx = -1;
			}
		}
		bool result = false;
		if (suffixIdx < 0)
		{
			result = false;
		}
		else if (key & 0x1) //There is suffix to check
		{
			const char nullByte = static_cast<char>(0);
			char *endSuffixIt = suffixes + endSuffixIdx;
			for (char *suffixIt = suffixes + suffixIdx; suffixIt < endSuffixIt; ++suffixIt)
			{
				char *wordIt = words + beginIdx + CHARSTOHASH; //Pointer to suffix of the word
				while (*suffixIt != nullByte && *wordIt != nullByte)
				{
					if (*suffixIt != *wordIt)
						break;
					++suffixIt;
					++wordIt;
				}
				if (*suffixIt == nullByte && *wordIt == nullByte)
				{
					result = true;
					break;
				}
				while (*suffixIt != nullByte)
				{
					++suffixIt;
				}
			}
		}
		else
		{
			result = true;
		}
		output[id] = result;
		id += threadsNum;
	}
}

template <class HASH, int B>
__global__ void kernel_find_words_v2(const int threadsNum, HASH* keysArray, int* indexesArray, int* sizeArray,
                                  const int rootIndex, const int height, char* suffixes, int suffixesSize,
                                  const int elementsNum, char* words, int* beginIndexes, bool* output)
{
	const int globalId = threadIdx.x + gridDim.x * blockIdx.x;
	constexpr int maxIndexesPerNode = B + 1;
	constexpr int maxKeysPerNode = B;
	int id = globalId;
	while (id < elementsNum)
	{
		const int beginIdx = beginIndexes[id];
		const HASH key = get_hash<HASH>(words, beginIdx);
		int currentHeight = 0;
		int node = rootIndex;
		//Inner nodes
		while (currentHeight < height)
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys;
			while (keys_begin + 1 != keys_end)
			{
				keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			if (*keys_begin <= key)
				++keys_begin;
			node = indexesArray[node * maxIndexesPerNode + keys_begin - (keysArray + node * maxKeysPerNode)];
			currentHeight += 1;
		}
		int suffixIdx, endSuffixIdx = -1;
		//Leaf level
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys;
			while (keys_begin + 1 != keys_end)
			{
				keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			keys = keys_begin;
			keys_begin = keysArray + node * maxKeysPerNode;
			keys_end = keys_begin + size;
			if (keys < keys_end && *keys == key)
			{
				const int indexInKeyArray = keys - keys_begin;
				suffixIdx = indexesArray[node * maxIndexesPerNode + indexInKeyArray];
				if (indexInKeyArray < size - 1) //Next element is in the same leaf
				{
					endSuffixIdx = indexesArray[node * maxIndexesPerNode + indexInKeyArray + 1];
				}
				else //Next element is in the next leaf
				{
					if (indexesArray[node * maxIndexesPerNode + maxIndexesPerNode - 1] != -1) //Next leaf exists
					{
						endSuffixIdx = indexesArray[(node + 1) * maxIndexesPerNode];
					}
					else //It is the last element in the last leaf
					{
						endSuffixIdx = suffixesSize;
					}
				}
			}
			else
			{
				suffixIdx = -1;
			}
		}
		bool result = false;
		if (suffixIdx < 0)
		{
			result = false;
		}
		else if (key & 0x1) //There is suffix to check
		{
			const char nullByte = static_cast<char>(0);
			char *endSuffixIt = suffixes + endSuffixIdx;
			for (char *suffixIt = suffixes + suffixIdx; suffixIt < endSuffixIt; ++suffixIt)
			{
				char *wordIt = words + beginIdx + chars_in_type<HASH>; //Pointer to suffix of the word
				while (*suffixIt != nullByte && *wordIt != nullByte)
				{
					if (*suffixIt != *wordIt)
						break;
					++suffixIt;
					++wordIt;
				}
				if (*suffixIt == nullByte && *wordIt == nullByte)
				{
					result = true;
					break;
				}
				while (*suffixIt != nullByte)
				{
					++suffixIt;
				}
			}
		}
		else
		{
			result = true;
		}
		output[id] = result;
		id += threadsNum;
	}
}

template <class HASH, int B>
__global__ void kernel_find_words_v3(const int threadsNum, HASH* keysArray, int* indexesArray, int* sizeArray,
                                  const int rootIndex, const int height, char* suffixes, int suffixesSize,
                                  const int elementsNum, char* words, int* beginIndexes, bool* output)
{
	const int globalId = GetGlobalId();
	const int maxIndexesPerNode = B + 1;
	const int maxKeysPerNode = B;
	int id = globalId;
	while (id < elementsNum)
	{
		const int beginIdx = beginIndexes[id];
		const HASH key = get_hash_v2<HASH>(words, beginIdx);
		int currentHeight = 0;
		int node = rootIndex;
		//Inner nodes
		while (currentHeight < height)
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys;
			while (keys_begin + 1 != keys_end)
			{
				keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			if (*keys_begin <= key)
				++keys_begin;
			node = indexesArray[node * maxIndexesPerNode + keys_begin - (keysArray + node * maxKeysPerNode)];
			currentHeight += 1;
		}
		int suffixIdx, endSuffixIdx = -1;
		//Leaf level
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys;
			while (keys_begin + 1 != keys_end)
			{
				keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			keys = keys_begin;
			keys_begin = keysArray + node * maxKeysPerNode;
			keys_end = keys_begin + size;
			if (keys < keys_end && *keys == key)
			{
				const int indexInKeyArray = keys - keys_begin;
				suffixIdx = indexesArray[node * maxIndexesPerNode + indexInKeyArray];
				if (indexInKeyArray < size - 1) //Next element is in the same leaf
				{
					endSuffixIdx = indexesArray[node * maxIndexesPerNode + indexInKeyArray + 1];
				}
				else //Next element is in the next leaf
				{
					if (indexesArray[node * maxIndexesPerNode + maxIndexesPerNode - 1] != -1) //Next leaf exists
					{
						endSuffixIdx = indexesArray[(node + 1) * maxIndexesPerNode];
					}
					else //It is the last element in the last leaf
					{
						endSuffixIdx = suffixesSize;
					}
				}
			}
			else
			{
				suffixIdx = -1;
			}
		}
		bool result = false;
		if (suffixIdx < 0)
		{
			result = false;
		}
		else if (key & 0x1) //There is suffix to check
		{
			const char nullByte = static_cast<char>(0);
			char *endSuffixIt = suffixes + endSuffixIdx;
			for (char *suffixIt = suffixes + suffixIdx; suffixIt < endSuffixIt; ++suffixIt)
			{
				char *wordIt = words + beginIdx + chars_in_type<HASH>; //Pointer to suffix of the word
				while (*suffixIt != nullByte && *wordIt != nullByte)
				{
					if (*suffixIt != *wordIt)
						break;
					++suffixIt;
					++wordIt;
				}
				if (*suffixIt == nullByte && *wordIt == nullByte)
				{
					result = true;
					break;
				}
				while (*suffixIt != nullByte)
				{
					++suffixIt;
				}
			}
		}
		else
		{
			result = true;
		}
		output[id] = result;
		id += threadsNum;
	}
}

template <class HASH, int B>
__global__ void kernel_find_words_v4(const int threadsNum, HASH* keysArray, int* indexesArray, int* sizeArray,
                                  const int rootIndex, const int height, char* suffixes, int suffixesSize,
                                  const int elementsNum, char* words, int* beginIndexes, bool* output)
{
	const int globalId = GetGlobalId();
	const int maxIndexesPerNode = B + 1;
	const int maxKeysPerNode = B;
	int id = globalId;
	while (id < elementsNum)
	{
		const int beginIdx = beginIndexes[id];
		const HASH key = get_hash_v3<HASH>(words, beginIdx);
		int currentHeight = 0;
		int node = rootIndex;
		//Inner nodes
		while (currentHeight < height)
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys;
			while (keys_begin + 1 != keys_end)
			{
				keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			if (*keys_begin <= key)
				++keys_begin;
			node = indexesArray[node * maxIndexesPerNode + keys_begin - (keysArray + node * maxKeysPerNode)];
			currentHeight += 1;
		}
		int suffixIdx, endSuffixIdx = -1;
		//Leaf level
		{
			const int size = sizeArray[node];
			const HASH *keys_begin = keysArray + node * maxKeysPerNode;
			const HASH *keys_end = keys_begin + size;
			const HASH *keys;
			while (keys_begin + 1 != keys_end)
			{
				keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			keys = keys_begin;
			keys_begin = keysArray + node * maxKeysPerNode;
			keys_end = keys_begin + size;
			if (keys < keys_end && *keys == key)
			{
				const int indexInKeyArray = keys - keys_begin;
				suffixIdx = indexesArray[node * maxIndexesPerNode + indexInKeyArray];
				if (indexInKeyArray < size - 1) //Next element is in the same leaf
				{
					endSuffixIdx = indexesArray[node * maxIndexesPerNode + indexInKeyArray + 1];
				}
				else //Next element is in the next leaf
				{
					if (indexesArray[node * maxIndexesPerNode + maxIndexesPerNode - 1] != -1) //Next leaf exists
					{
						endSuffixIdx = indexesArray[(node + 1) * maxIndexesPerNode];
					}
					else //It is the last element in the last leaf
					{
						endSuffixIdx = suffixesSize;
					}
				}
			}
			else
			{
				suffixIdx = -1;
			}
		}
		bool result = false;
		if (suffixIdx < 0)
		{
			result = false;
		}
		else if (key & 0x1) //There is suffix to check
		{
			const char nullByte = static_cast<char>(0);
			char *endSuffixIt = suffixes + endSuffixIdx;
			for (char *suffixIt = suffixes + suffixIdx; suffixIt < endSuffixIt; ++suffixIt)
			{
				char *wordIt = words + beginIdx + chars_in_type<HASH>; //Pointer to suffix of the word
				while (*suffixIt != nullByte && *wordIt != nullByte)
				{
					if (*suffixIt != *wordIt)
						break;
					++suffixIt;
					++wordIt;
				}
				if (*suffixIt == nullByte && *wordIt == nullByte)
				{
					result = true;
					break;
				}
				while (*suffixIt != nullByte)
				{
					++suffixIt;
				}
			}
		}
		else
		{
			result = true;
		}
		output[id] = result;
		id += threadsNum;
	}
}

template<typename HASH>
union reuse_pointer
{
	char* _char;
	int* _int;
	bool* _bool;
	HASH* _hash;
	void* _any;
};

template <class HASH, int B>
__global__ void kernel_find_words_v5(const int threadsNum, HASH* keysArray, int* indexesArray, int* sizeArray,
                                  const int rootIndex, const int height, char* suffixes, int suffixesSize,
                                  const int elementsNum, reuse_pointer<HASH> words, reuse_pointer<HASH> beginIndexes, reuse_pointer<HASH> output)
{
	__shared__ void* help[672];
	const int maxIndexesPerNode = B + 1;
	const int maxKeysPerNode = B;
	int id = GetGlobalIdSlim();
	while (id < elementsNum)
	{
		const HASH key = get_hash_v2<HASH>(words._char, beginIndexes._int[id]);
		{
			const int local = GetLocalIdSlim();
			help[local] = words._any;
			//help[local + 1024] = beginIndexes._any;
			//help[local + 2048] = output._any;
		}
		int suffixIdx, endSuffixIdx = -1;
		{
#define keys_begin words._hash
//#define keys_end beginIndexes._hash
//#define keys output._hash
			HASH *keys, *keys_end;
			int node = rootIndex;
			{
				int currentHeight = 0;
				//Inner nodes
				while (currentHeight < height)
				{
					keys_begin = keysArray + node * maxKeysPerNode;
					keys_end = keys_begin + sizeArray[node];
					while (keys_begin + 1 != keys_end)
					{
						keys = keys_begin + ((keys_end - keys_begin) >> 1);
						if (*keys <= key)
							keys_begin = keys;
						else
							keys_end = keys;
					}
					if (*keys_begin <= key)
						keys_begin = keys_begin + 1;
					node = indexesArray[node * maxIndexesPerNode + keys_begin - (keysArray + node * maxKeysPerNode)];
					currentHeight += 1;
				}
			}
			//Leaf level
			{
				const int size = sizeArray[node];
				keys_begin = keysArray + node * maxKeysPerNode;
				keys_end = keys_begin + size;
				while (keys_begin + 1 != keys_end)
				{
					keys = keys_begin + ((keys_end - keys_begin) >> 1);
					if (*keys <= key)
						keys_begin = keys;
					else
						keys_end = keys;
				}
				keys = keys_begin;
				keys_begin = keysArray + node * maxKeysPerNode;
				keys_end = keys_begin + size;
				if (keys < keys_end && *keys == key)
				{
					suffixIdx = indexesArray[node * maxIndexesPerNode + keys - keys_begin];
					if (keys - keys_begin < size - 1) //Next element is in the same leaf
					{
						endSuffixIdx = indexesArray[node * maxIndexesPerNode + keys - keys_begin + 1];
					}
					else //Next element is in the next leaf
					{
						if (indexesArray[node * maxIndexesPerNode + maxIndexesPerNode - 1] != -1) //Next leaf exists
						{
							endSuffixIdx = indexesArray[(node + 1) * maxIndexesPerNode];
						}
						else //It is the last element in the last leaf
						{
							endSuffixIdx = suffixesSize;
						}
					}
				}
				else
				{
					suffixIdx = -1;
				}
			}
#undef keys_begin
		}
		{
			const int local = GetLocalIdSlim();
			words._char = reinterpret_cast<char*>(help[local]);
			//beginIndexes._int = reinterpret_cast<int*>(help[local + 1024]);
			//output._bool = reinterpret_cast<bool*>(help[local + 2048]);
		}
		if (suffixIdx < 0)
		{
			output._bool[id] = false;
		}
		else if (key & 0x1) //There is suffix to check
		{
			char *endSuffixIt = suffixes + endSuffixIdx;
			for (char *suffixIt = suffixes + suffixIdx; suffixIt < endSuffixIt; ++suffixIt)
			{
				char *wordIt = words._char + beginIndexes._int[id] + chars_in_type<HASH>; //Pointer to suffix of the word
				while (*suffixIt != static_cast<char>(0) && *wordIt != static_cast<char>(0))
				{
					if (*suffixIt != *wordIt)
						break;
					++suffixIt;
					++wordIt;
				}
				if (*suffixIt == static_cast<char>(0) && *wordIt == static_cast<char>(0))
				{
					output._bool[id] = true;
					break;
				}
				while (*suffixIt != static_cast<char>(0))
				{
					++suffixIt;
				}
			}
		}
		else
		{
			output._bool[id] = true;
		}
		id += threadsNum;
	}
}

template <class HASH, int B>
__global__ void kernel_find_words_v6(const int threadsNum, HASH* keysArray, int* indexesArray, int* sizeArray,
                                  const int rootIndex, const int height, char* suffixes, int suffixesSize,
                                  const int elementsNum, char* words, int* beginIndexes, bool* output)
{
	const int maxIndexesPerNode = B + 1;
	const int maxKeysPerNode = B;
	int id = GetGlobalIdSlim();
	const HASH key = get_hash_v2<HASH>(words, beginIndexes[id]);
	int suffixIdx, endSuffixIdx = -1;
	{
		HASH *keys, *keys_end, *keys_begin;
		int node = rootIndex;
		{
			int currentHeight = 0;
			//Inner nodes
			while (currentHeight < height)
			{
				keys_begin = keysArray + node * maxKeysPerNode;
				keys_end = keys_begin + sizeArray[node];
				while (keys_begin + 1 != keys_end)
				{
					keys = keys_begin + ((keys_end - keys_begin) >> 1);
					if (*keys <= key)
						keys_begin = keys;
					else
						keys_end = keys;
				}
				if (*keys_begin <= key)
					keys_begin = keys_begin + 1;
				node = indexesArray[node * maxIndexesPerNode + keys_begin - (keysArray + node * maxKeysPerNode)];
				currentHeight += 1;
			}
		}
		//Leaf level
		{
			const int size = sizeArray[node];
			keys_begin = keysArray + node * maxKeysPerNode;
			keys_end = keys_begin + size;
			while (keys_begin + 1 != keys_end)
			{
				keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			keys = keys_begin;
			keys_begin = keysArray + node * maxKeysPerNode;
			keys_end = keys_begin + size;
			if (keys < keys_end && *keys == key)
			{
				suffixIdx = indexesArray[node * maxIndexesPerNode + keys - keys_begin];
				if (keys - keys_begin < size - 1) //Next element is in the same leaf
				{
					endSuffixIdx = indexesArray[node * maxIndexesPerNode + keys - keys_begin + 1];
				}
				else //Next element is in the next leaf
				{
					if (indexesArray[node * maxIndexesPerNode + maxIndexesPerNode - 1] != -1) //Next leaf exists
					{
						endSuffixIdx = indexesArray[(node + 1) * maxIndexesPerNode];
					}
					else //It is the last element in the last leaf
					{
						endSuffixIdx = suffixesSize;
					}
				}
			}
			else
			{
				suffixIdx = -1;
			}
		}
	}
	if (suffixIdx < 0)
	{
		output[id] = false;
	}
	else if (key & 0x1) //There is suffix to check
	{
		char *endSuffixIt = suffixes + endSuffixIdx;
		for (char *suffixIt = suffixes + suffixIdx; suffixIt < endSuffixIt; ++suffixIt)
		{
			char *wordIt = words+ beginIndexes[id] + chars_in_type<HASH>; //Pointer to suffix of the word
			while (*suffixIt != static_cast<char>(0) && *wordIt != static_cast<char>(0))
			{
				if (*suffixIt != *wordIt)
					break;
				++suffixIt;
				++wordIt;
			}
			if (*suffixIt == static_cast<char>(0) && *wordIt == static_cast<char>(0))
			{
				output[id] = true;
				break;
			}
			while (*suffixIt != static_cast<char>(0))
			{
				++suffixIt;
			}
		}
	}
	else
	{
		output[id] = true;
	}
}

template <class HASH, int B>
__device__ __inline__ void get_leaf_node(const HASH* keysArray, const int* indexesArray, const int* sizeArray, const int height, const HASH key, int& node)
{
	const int max_indexes_per_node = B + 1;
	const int max_keys_per_node = B;

	for (int currentHeight = 0; currentHeight < height; currentHeight++)
	{
		const int size = sizeArray[node];
		int start_index = node * max_keys_per_node;
		int end_index = start_index + size;

		while (start_index + 1 != end_index)
		{
			const int mid = end_index + start_index >> 1;
			if (keysArray[mid] <= key)
				start_index = mid;
			else
				end_index = mid;
		}

		if (keysArray[start_index] <= key)
			++start_index;

		node = indexesArray[start_index + node * (max_indexes_per_node - max_keys_per_node)];
	}
}

template <class HASH, int B>
__device__ __inline__ void find_suffix_indices(const HASH* keysArray, const int* indexesArray, const int* sizeArray,
	const int suffixesSize, const HASH key, int node, int& suffix_idx, int& end_suffix_idx)
{
	const int max_indexes_per_node = B + 1;
	const int max_keys_per_node = B;

	const int size = sizeArray[node];
	int start_index = node * max_keys_per_node;
	int end_index = start_index + size;

	int index;

	while (start_index + 1 != end_index)
	{
		index = end_index + start_index >> 1;
		if (keysArray[index] <= key)
			start_index = index;
		else
			end_index = index;
	}

	index = start_index;
	start_index = node * max_keys_per_node;
	end_index = start_index + size;

	if (index >= end_index || keysArray[index] != key)
		return;

	const int index_in_key_array = index - start_index;
	suffix_idx = indexesArray[node * max_indexes_per_node + index_in_key_array];

	if (index_in_key_array < size - 1)
		//Next element is in the same leaf
		end_suffix_idx = indexesArray[node * max_indexes_per_node + index_in_key_array + 1];
	else
		//Next element is in the next leaf
		if (indexesArray[node * max_indexes_per_node + max_indexes_per_node - 1] != -1)
			//Next leaf exists
			end_suffix_idx = indexesArray[(node + 1) * max_indexes_per_node];
		else
			//It is the last element in the last leaf
			end_suffix_idx = suffixesSize;
}

template <class HASH>
__device__ __inline__ bool check_suffix(const char* suffixes, const char* words, const int begin_idx, int& suffix_idx, const int end_suffix_idx)
{
	const auto null_byte = static_cast<char>(0);
	for (; suffix_idx < end_suffix_idx; ++suffix_idx)
	{
		int word_suffix_index = begin_idx + chars_in_type<HASH>;;
		while (true)
		{
			const auto dict_char = suffixes[suffix_idx];
			const auto word_char = words[word_suffix_index];
			if (dict_char == null_byte && word_char == null_byte)
				return true;

			if (dict_char != word_char || dict_char == null_byte || word_char == null_byte)
				break;

			++word_suffix_index;
			++suffix_idx;
		}

		while (suffixes[suffix_idx] != null_byte)
			++suffix_idx;
	}
	return false;
}

template <class HASH, int B>
__global__ void kernel_find_words_v7(const int threadsNum, const HASH* keysArray, const int* indexesArray, const int* sizeArray,
								const int rootIndex, const int height, const char* suffixes, const int suffixesSize, const int elementsNum, const char* words,
								const int* beginIndexes, bool* output)
{
	const int id = GetGlobalId();
	if (id >= elementsNum)
		return;

	const int beginIdx = beginIndexes[id];
	const HASH key = get_hash<HASH>(words, beginIdx);
	int node = rootIndex;

	get_leaf_node<HASH, B>(keysArray, indexesArray, sizeArray, height, key, node);

	int suffix_idx;
	int end_suffix_idx = -1;
	find_suffix_indices<HASH, B>(keysArray, indexesArray, sizeArray, suffixesSize, key, node, suffix_idx, end_suffix_idx);

	if (end_suffix_idx == -1)
		return;//false

	if (!(key & 0x1))
	{
		output[id] = true;
		return;
	}

	output[id] = check_suffix<HASH>(suffixes, words, beginIdx, suffix_idx, end_suffix_idx);
}

#pragma endregion

#pragma region kernel_version_selectors
template <typename HASH, int B, int Version>
struct kernel_version_selector
{
	static_assert(Version - Version != 0, "Selected version of kernel does not exist.");
	static constexpr decltype(kernel_find_words_v1<HASH, B>) *kernel = nullptr;
	static constexpr int wordsAlignment = 1;
};

template <typename HASH, int B>
struct kernel_version_selector<HASH, B, 1>
{
	static constexpr decltype(kernel_find_words_v1<HASH, B>) *kernel = kernel_find_words_v1<HASH, B>;
	static constexpr int wordsAlignment = 1;
};

template <typename HASH, int B>
struct kernel_version_selector<HASH, B, 2>
{
	static constexpr decltype(kernel_find_words_v1<HASH, B>) *kernel = kernel_find_words_v2<HASH, B>;
	static constexpr int wordsAlignment = 1;
};

template <typename HASH, int B>
struct kernel_version_selector<HASH, B, 3>
{
	static constexpr decltype(kernel_find_words_v1<HASH, B>) *kernel = kernel_find_words_v3<HASH, B>;
	static constexpr int wordsAlignment = sizeof(uint32_t);
};

template <typename HASH, int B>
struct kernel_version_selector<HASH, B, 4>
{
	static constexpr decltype(kernel_find_words_v1<HASH, B>) *kernel = kernel_find_words_v4<HASH, B>;
	static constexpr int wordsAlignment = sizeof(uint4);
};

template <typename HASH, int B>
struct kernel_version_selector<HASH, B, 5>
{
	static constexpr decltype(kernel_find_words_v1<HASH, B>) *kernel = reinterpret_cast<decltype(kernel_find_words_v1<HASH, B>)*>(kernel_find_words_v5<HASH, B>);
	static constexpr int wordsAlignment = sizeof(uint32_t);
};

template <typename HASH, int B>
struct kernel_version_selector<HASH, B, 6>
{
	static constexpr decltype(kernel_find_words_v1<HASH, B>) *kernel = kernel_find_words_v6<HASH, B>;
	static constexpr int wordsAlignment = sizeof(uint32_t);
};

template <typename HASH, int B>
struct kernel_version_selector<HASH, B, 7>
{
	static constexpr decltype(kernel_find_words_v1<HASH, B>) *kernel = reinterpret_cast<decltype(kernel_find_words_v1<HASH, B>)*>(kernel_find_words_v7<HASH, B>);
	static constexpr int wordsAlignment = 1;
};

#pragma endregion kernel_version_selectors

template <class HASH, int B>
class bplus_tree_gpu
{
	float m_elapsedTime;
public:
	char* suffixes;
	int suffixesSize;
	int* indexesArray;
	HASH* keysArray;
	int* sizeArray;
	HASH* minArray;
	int reservedNodes;
	int usedNodes;
	int rootNodeIndex;
	int height;
protected:
	void create_tree(const HASH* hashes, const int* values, int size, const char* suffixes, int suffixesLength);
	static int needed_nodes(int elemNum);
public:
	bplus_tree_gpu(bplus_tree_gpu<HASH, B>& gTree);
	bplus_tree_gpu(const HASH* hashes, const int* values, int size, const char *suffixes, int suffixesLength);
	~bplus_tree_gpu();

	bool exist(HASH key);
	std::vector<bool> exist(HASH* keys, int size);

	template<int Version>
	std::vector<bool> exist_word(const char* words, int wordsSize, const int* beginIndexes, int indexesSize, float &preTime, float &execTime, float &postTime);

	int get_value(HASH key);
	std::vector<int> get_value(HASH* keys, int size);

	bool insert(HASH key, int value);

	void bulk_insert(HASH* keys, int* values, int size);

	int get_height();

	float last_gpu_time() const;
};

template <class HASH, int B>
void bplus_tree_gpu<HASH, B>::create_tree(const HASH* hashes, const int* values, int size, const char* suffixes, int suffixesLength)
{
	height = 0;
	const int elementNum = size; //Number of hashes
	reservedNodes = needed_nodes(size);
	HASH* d_hashes;
	int* d_values;
	suffixesSize = suffixesLength;
	cudaEvent_t startEvent, stopEvent;
	gpuErrchk(cudaEventCreate(&startEvent));
	gpuErrchk(cudaEventCreate(&stopEvent));
	gpuErrchk(cudaEventRecord(startEvent));
	gpuErrchk(cudaMalloc(&(this->suffixes), suffixesLength * sizeof(char)));
	gpuErrchk(cudaMalloc(&indexesArray, reservedNodes * sizeof(HASH) * (B + 1)));
	gpuErrchk(cudaMalloc(&keysArray, reservedNodes * sizeof(HASH) * B));
	gpuErrchk(cudaMalloc(&sizeArray, reservedNodes * sizeof(int)));
	gpuErrchk(cudaMalloc(&minArray, reservedNodes * sizeof(HASH)));
	gpuErrchk(cudaMalloc(&d_hashes, size * sizeof(HASH)));
	gpuErrchk(cudaMalloc(&d_values, size* sizeof(int)));
	gpuErrchk(cudaMemcpy(this->suffixes, suffixes, sizeof(char) * suffixesLength, cudaMemcpyHostToDevice)); //Suffixes are copied to this->suffixes
	gpuErrchk(cudaMemcpy(d_hashes, hashes, sizeof(HASH) * size, cudaMemcpyHostToDevice)); //Keys are copied to d_hashes
	gpuErrchk(cudaMemcpy(d_values, values, sizeof(int) * size, cudaMemcpyHostToDevice)); //Values are copied to d_values

	int blocksNum = elementNum <= 32 ? 1 : 2;
	int threadsNum = elementNum <= 32 ? 32 : std::min(elementNum / 2, 1024);
	kernel_create_leafs<HASH, B> kernel_init(blocksNum, threadsNum) (threadsNum, elementNum, d_hashes, d_values, keysArray,
	                                                                 sizeArray, indexesArray, minArray);
	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaFree(d_hashes));
	gpuErrchk(cudaFree(d_values));
	int lastCreated = std::max(1, elementNum * 2 / B);
	int beginIndex = 0;
	int endIndex = lastCreated;
	while (lastCreated != 1)
	{
		height += 1;
		blocksNum = lastCreated <= 32 ? 1 : 2;
		threadsNum = lastCreated <= 32 ? 32 : std::min(lastCreated / 2, 1024);
		kernel_create_next_layer<HASH, B> kernel_init(blocksNum, threadsNum) (
			threadsNum, beginIndex, endIndex, indexesArray, keysArray, sizeArray, minArray);
		gpuErrchk(cudaGetLastError());
		lastCreated = std::max(1, lastCreated / (B / 2 + 1));
		beginIndex = endIndex;
		endIndex = endIndex + lastCreated;
	}
	gpuErrchk(cudaEventRecord(stopEvent));
	gpuErrchk(cudaEventSynchronize(stopEvent));
	gpuErrchk(cudaEventElapsedTime(&m_elapsedTime, startEvent, stopEvent));
	rootNodeIndex = endIndex - 1;
	usedNodes = endIndex;
}

template <class HASH, int B>
bplus_tree_gpu<HASH, B>::bplus_tree_gpu(bplus_tree_gpu<HASH, B>& gTree)
{
	reservedNodes = gTree.reservedNodes;
	usedNodes = gTree.usedNodes;
	rootNodeIndex = gTree.rootNodeIndex;
	height = gTree.height;
	gpuErrchk(cudaMalloc(&indexesArray, reservedNodes * sizeof(HASH) * (B + 1)));
	gpuErrchk(cudaMalloc(&keysArray, reservedNodes * sizeof(HASH) * B));
	gpuErrchk(cudaMalloc(&sizeArray, reservedNodes * sizeof(int)));
	gpuErrchk(cudaMemcpy(indexesArray, gTree.indexesArray, reservedNodes * sizeof(HASH) * (B + 1), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(keysArray, gTree.keysArray, reservedNodes * sizeof(HASH) * B, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(sizeArray, gTree.sizeArray, reservedNodes * sizeof(int), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(minArray, gTree.minArray, reservedNodes * sizeof(HASH), cudaMemcpyDeviceToDevice));
}

template <class HASH, int B>
bplus_tree_gpu<HASH, B>::bplus_tree_gpu(const HASH* hashes, const int* values, int size, const char *suffixes, int suffixesLength)
{
	create_tree(hashes, values, size, suffixes, suffixesLength);
}

template <class HASH, int B>
bplus_tree_gpu<HASH, B>::~bplus_tree_gpu()
{
	gpuErrchk(cudaFree(indexesArray));
	gpuErrchk(cudaFree(keysArray));
	gpuErrchk(cudaFree(sizeArray));
	gpuErrchk(cudaFree(minArray));
	gpuErrchk(cudaFree(suffixes));
}

template <class HASH, int B>
bool bplus_tree_gpu<HASH, B>::exist(HASH key)
{
	return exist(&key, 1)[0];
}

template <class HASH, int B>
std::vector<bool> bplus_tree_gpu<HASH, B>::exist(HASH* keys, int size)
{
	const int elementNum = size;
	HASH* d_keys;
	bool *output = new bool[size];
	bool* d_output;
	gpuErrchk(cudaMalloc(&d_keys, size * sizeof(HASH)));
	gpuErrchk(cudaMalloc(&d_output, size * sizeof(bool)));
	gpuErrchk(cudaMemcpy(d_keys, keys, size * sizeof(HASH), cudaMemcpyHostToDevice));

	const int blocksNum = elementNum <= 32 ? 1 : 2;
	const int threadsNum = elementNum <= 32 ? 32 : std::min(elementNum / 2, 1024);
	kernel_get_value<HASH, B> kernel_init(blocksNum, threadsNum) (threadsNum, elementNum, keysArray, sizeArray, indexesArray, d_keys, height, rootNodeIndex, d_output);
	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaMemcpy(output, d_output, size * sizeof(bool), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_output));
	gpuErrchk(cudaFree(d_keys));
	return std::vector<bool>(output, output + size);
}

template <class HASH, int B>
template <int Version>
std::vector<bool> bplus_tree_gpu<HASH, B>::exist_word(const char* words, int wordsSize, const int* beginIndexes, int indexesSize, float &preTime, float &execTime, float &postTime)
{
	const int elementNum = indexesSize;
	char *d_words;
	int *d_indexes;
	bool *d_output;
	cudaDeviceProp props;
	gpuErrchk(cudaGetDeviceProperties(&props, 0));
	const int threadsNum = (Version == 6 || Version == 7) ? 128 :
		Version == 5 ? 672 :
		1024;
	const int blocksNum = (Version == 7 || Version == 6) ? std::ceil(static_cast<float>(elementNum) / threadsNum) : std::round(props.multiProcessorCount * 2048.0 / threadsNum);
	cudaEvent_t startEvent, preEvent, execEvent, postEvent;
	gpuErrchk(cudaEventCreate(&startEvent));
	gpuErrchk(cudaEventCreate(&preEvent));
	gpuErrchk(cudaEventCreate(&execEvent));
	gpuErrchk(cudaEventCreate(&postEvent));
	gpuErrchk(cudaEventRecord(startEvent))
	gpuErrchk(cudaMalloc(&d_indexes, indexesSize * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_words, wordsSize * sizeof(char)));
	gpuErrchk(cudaMalloc(&d_output, indexesSize * sizeof(bool)));
	gpuErrchk(cudaMemcpy(d_words, words, wordsSize * sizeof(char), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_indexes, beginIndexes, indexesSize * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaEventRecord(preEvent));
	kernel_version_selector<HASH, B, Version>::kernel kernel_init(blocksNum, threadsNum)
		(threadsNum * blocksNum, keysArray, indexesArray, sizeArray,
			rootNodeIndex, height, suffixes, suffixesSize,
			elementNum, d_words, d_indexes, d_output);
	gpuErrchk(cudaEventRecord(execEvent));

	gpuErrchk(cudaGetLastError());

	bool *output = new bool[elementNum];
	gpuErrchk(cudaMemcpy(output, d_output, elementNum * sizeof(bool), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_words));
	gpuErrchk(cudaFree(d_indexes));
	gpuErrchk(cudaFree(d_output));
	gpuErrchk(cudaEventRecord(postEvent));
	gpuErrchk(cudaEventSynchronize(postEvent));
	gpuErrchk(cudaEventElapsedTime(&preTime, startEvent, preEvent));
	gpuErrchk(cudaEventElapsedTime(&execTime, preEvent, execEvent));
	gpuErrchk(cudaEventElapsedTime(&postTime, execEvent, postEvent));
	std::vector<bool> v(output, output + elementNum);
	delete[] output;
	return v;
}

template <class HASH, int B>
int bplus_tree_gpu<HASH, B>::get_value(HASH key)
{
	return get_value(&key, 1)[0];
}

template <class HASH, int B>
std::vector<int> bplus_tree_gpu<HASH, B>::get_value(HASH* keys, int size)
{
	const int elementNum = size;
	HASH* d_keys;
	std::vector<int> output(size);
	int* d_output;
	gpuErrchk(cudaMalloc(&d_keys, size * sizeof(HASH)));
	gpuErrchk(cudaMalloc(&d_output, size * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_keys, keys, size * sizeof(HASH), cudaMemcpyHostToDevice));

	const int blocksNum = elementNum <= 32 ? 1 : 2;
	const int threadsNum = elementNum <= 32 ? 32 : std::min(elementNum / 2, 1024);
	kernel_get_value<HASH, B> kernel_init(blocksNum, threadsNum) (threadsNum, elementNum, keysArray, sizeArray, indexesArray, d_keys, height, rootNodeIndex, d_output);
	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaMemcpy(output.data(), d_output, size * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_output));
	gpuErrchk(cudaFree(d_keys));
	return output;
}

template <class HASH, int B>
bool bplus_tree_gpu<HASH, B>::insert(HASH key, int value)
{
	throw not_implemented();
}

template <class HASH, int B>
void bplus_tree_gpu<HASH, B>::bulk_insert(HASH* keys, int* values, int size)
{
	throw not_implemented();
}

template <class HASH, int B>
int bplus_tree_gpu<HASH, B>::get_height()
{
	return height;
}

template <class HASH, int B>
float bplus_tree_gpu<HASH, B>::last_gpu_time() const
{
	return m_elapsedTime;
}

template <class HASH, int B>
int bplus_tree_gpu<HASH, B>::needed_nodes(int elemNum)
{
	if (elemNum < B)
		return 1;
	int pages = elemNum * 2 / B;
	elemNum = pages;
	while (elemNum > B + 1)
	{
		elemNum = elemNum / (B / 2 + 1);
		pages += elemNum;
	}
	pages += 1;
	return pages;
}
