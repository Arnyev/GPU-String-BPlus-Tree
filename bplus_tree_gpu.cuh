#pragma once
#include <vector>
#include "bplus_tree.h"
#include "gpu_helper.cuh"
#include "not_implemented.h"
#include <cassert>

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

template<class HASH, int B>
__global__ void kernel_create_next_layer(const int threadsNum, const int beginIndex, const int endIndex, int* indexArray, HASH* keysArray, int* sizeArray, int *minArray, int* output)
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
	//Output
	if (globalId == 0)
	{
		reinterpret_cast<output_create_next_layer*>(output)->lastUsedIndex = toCreate + endIndex;
		reinterpret_cast<output_create_next_layer*>(output)->isRoot = toCreate == 1;
	}
}

template<class HASH, int B>
__global__ void kernel_create_leafs(const int threadsNum, const int elementNum, HASH* hashesArray, int* valueArray, HASH* keysArray, int* sizeArray, int* indexesArray, int* minArray, int* output)
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
	//Filling output
	if (globalId == 0)
	{
		reinterpret_cast<output_create_leafs*>(output)->rootNodeIndex = 0;
		reinterpret_cast<output_create_leafs*>(output)->usedNodes = bottomPages;
		reinterpret_cast<output_create_leafs*>(output)->isOnlyRoot = bottomPages == 1 ? 1 : 0;
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
					output[id] = -1;
			}
		}
		id += threadsNum;
	}
}

template <class HASH, int B>
class bplus_tree_gpu : public bplus_tree<HASH, B>
{
public:
	int* indexesArray;
	HASH* keysArray;
	int* sizeArray;
	int* minArray;
	int reservedNodes;
	int usedNodes;
	int rootNodeIndex;
	int height;
protected:
	void create_tree(HASH* hashes, int* values, int size) override;
public:
	bplus_tree_gpu(bplus_tree_gpu<HASH, B>& gTree);
	bplus_tree_gpu(HASH* hashes, int* values, int size);
	~bplus_tree_gpu();

	bool exist(HASH key) override;
	std::vector<bool> exist(HASH* keys, int size) override;

	int get_value(HASH key) override;
	std::vector<int> get_value(HASH* keys, int size) override;

	bool insert(HASH key, int value) override;

	void bulk_insert(HASH* keys, int* values, int size) override;

	int get_height() override;
};

template <class HASH, int B>
void bplus_tree_gpu<HASH, B>::create_tree(HASH* hashes, int* values, int size)
{
	height = 0;
	int elementNum = size; //Number of hashes
	reservedNodes = needed_nodes(size);
	HASH* d_hashes;
	int* d_output;
	int* d_values;
	output_create_leafs h_output_create_leafs;
	gpuErrchk(cudaMalloc(&indexesArray, reservedNodes * sizeof(HASH) * (B + 1)));
	gpuErrchk(cudaMalloc(&keysArray, reservedNodes * sizeof(HASH) * B));
	gpuErrchk(cudaMalloc(&sizeArray, reservedNodes * sizeof(int)));
	gpuErrchk(cudaMalloc(&minArray, reservedNodes * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_hashes, size * sizeof(HASH)));
	gpuErrchk(cudaMalloc(&d_values, size* sizeof(int)));
	gpuErrchk(cudaMalloc(&d_output, sizeof(output_create_leafs)));

	gpuErrchk(cudaMemcpy(d_hashes, hashes, sizeof(HASH) * size, cudaMemcpyHostToDevice)); //Keys are copied to d_hashes
	gpuErrchk(cudaMemcpy(d_values, values, sizeof(int) * size, cudaMemcpyHostToDevice)); //Values are copied to d_values

	int threadsNum = elementNum < 1024 ? elementNum : 1024;
	int blocksNum = elementNum < 1024 ? 1 : static_cast<int>(std::ceil(elementNum / 1024.f));
	kernel_create_leafs<HASH, B> kernel_init(threadsNum, blocksNum) (threadsNum, elementNum, d_hashes, d_values, keysArray,
	                                                                 sizeArray, indexesArray, minArray, d_output);
	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaMemcpy(&h_output_create_leafs, d_output, sizeof(output_create_leafs), cudaMemcpyDeviceToHost));
	//Exctracting output
	gpuErrchk(cudaFree(d_hashes));
	gpuErrchk(cudaFree(d_output));
	int beginIndex = 0;
	int endIndex = h_output_create_leafs.usedNodes;
	bool isRoot = h_output_create_leafs.isOnlyRoot != 0;
	if (!isRoot)
	{
		output_create_next_layer h_output_create_next_layer;
		gpuErrchk(cudaMalloc(&d_output, sizeof(output_create_next_layer)));
		while (!isRoot)
		{
			height += 1;
			kernel_create_next_layer<HASH, B> kernel_init(threadsNum, blocksNum) (
				threadsNum, beginIndex, endIndex, indexesArray, keysArray, sizeArray, minArray, d_output);
			gpuErrchk(cudaGetLastError());
			gpuErrchk(cudaMemcpy(&h_output_create_next_layer, d_output, sizeof(output_create_next_layer), cudaMemcpyDeviceToHost)
			); //Exctracting output
			beginIndex = endIndex;
			endIndex = h_output_create_next_layer.lastUsedIndex;
			isRoot = h_output_create_next_layer.isRoot != 0;
		}
		gpuErrchk(cudaFree(d_output));
	}
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
	gpuErrchk(cudaMemcpy(minArray, gTree.minArray, reservedNodes * sizeof(int), cudaMemcpyDeviceToDevice));
}

template <class HASH, int B>
bplus_tree_gpu<HASH, B>::bplus_tree_gpu(HASH* hashes, int* values, int size)
{
	create_tree(hashes, values, size);
}

template <class HASH, int B>
bplus_tree_gpu<HASH, B>::~bplus_tree_gpu()
{
	gpuErrchk(cudaFree(indexesArray));
	gpuErrchk(cudaFree(keysArray));
	gpuErrchk(cudaFree(sizeArray));
	gpuErrchk(cudaFree(minArray));
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

	const int threadsNum = elementNum < 1024 ? elementNum : 1024;
	const int blocksNum = elementNum < 1024 ? 1 : static_cast<int>(std::ceil(elementNum / 1024.f));
	kernel_get_value<HASH, B> kernel_init(threadsNum, blocksNum) (threadsNum, elementNum, keysArray, sizeArray, indexesArray, d_keys, height, rootNodeIndex, d_output);
	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaMemcpy(output, d_output, size * sizeof(bool), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_output));
	gpuErrchk(cudaFree(d_keys));
	return std::vector<bool>(output, output + size);
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

	const int threadsNum = elementNum < 1024 ? elementNum : 1024;
	const int blocksNum = elementNum < 1024 ? 1 : static_cast<int>(std::ceil(elementNum / 1024.f));
	kernel_get_value<HASH, B> kernel_init(threadsNum, blocksNum) (threadsNum, elementNum, keysArray, sizeArray, indexesArray, d_keys, height, rootNodeIndex, d_output);
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
	//Assumption that new and old keys are unique
	//TODO merge keys and values
	HASH* newKeys;
	int* newValues;
	int newSize;
	create_tree(newKeys, newValues, newSize);
}

template <class HASH, int B>
int bplus_tree_gpu<HASH, B>::get_height()
{
	return height;
}
