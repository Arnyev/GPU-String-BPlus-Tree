#pragma once
#include <vector>
#include "bplus_tree.h"
#include "gpu_helper.cuh"
#include "not_implemented.h"

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
__global__ void kernel_create_next_layer(const int threadsNum, const int beginIndex, const int endIndex, int* indexArray, HASH* keysArray, int* sizeArray, int* output)
{
	const int globalId = GetGlobalId();
	const int minIndexesPerNode = B / 2 + 1;
	const int maxIndexesPerNode = B + 1;
	const int minKeysPerNode = B / 2;
	const int maxKeysPerNode = B;
	int createdNodes = endIndex - beginIndex; //How many nodes were last time created
	//Creation of new layer
	int toCreate = createdNodes / (B / 2 + 1); //How many nodes will be created
	//In each node there will be at least B / 2 keys and B / 2 + 1 indexes to lower layer nodes
	int id = globalId;
	while (id < createdNodes)
	{
		//Each thread serve one node from previous layer
		const int relativeNode = id / minIndexesPerNode; //Relative number of a current node
		const bool isMerge = relativeNode == toCreate && toCreate > 1; //Is current value gonna be merge to previous node
		const int absoluteNode = endIndex + relativeNode + (isMerge ? -1 : 0);
		/* Absolute number of a current node. If it is not root node (toCreate > 1) and relative number of a node exceeds number of nodes to create
		 * (relativeNode == toCreate, because realativeNode starts from 0), then absolute index is decreased by 1 due to merge of the nodes.
		 */
		const int nodeOffset = id - relativeNode * minIndexesPerNode + (isMerge ? minIndexesPerNode : 0); //Offset on the current node
		const int lowerNodeIndex = beginIndex + id; //Index of a lower node
		//Writes indexes from one level bellow to nodes
		{
			const int target = absoluteNode * maxIndexesPerNode + nodeOffset;
			indexArray[target] = lowerNodeIndex;
		}
		//Writes keys from one level bellow to nodes
		if (nodeOffset != 0)
		{
			const int target = absoluteNode * maxKeysPerNode + nodeOffset - 1;
			const int source = lowerNodeIndex * maxKeysPerNode;
			keysArray[target] = keysArray[source];
		}
		id += threadsNum;
	}
	//Filling size of nodes
	id = globalId;
	while (id < toCreate)
	{
		const int leftElements = createdNodes - id * (minKeysPerNode + 1);
		sizeArray[endIndex + id] = leftElements <= maxIndexesPerNode ? leftElements - 1 : minKeysPerNode;
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
__global__ void kernel_create_leafs(const int threadsNum, const int elementNum, HASH* hashesArray, int* valueArray, HASH* keysArray, int* sizeArray, int* indexesArray, int* output)
{
	const int globalId = GetGlobalId();

	int currentNode = 0; //Index of first not initilize node
	int bottomPages = elementNum * 2 / B;
	int elementsOnLastPage = elementNum - (bottomPages - 1) * B / 2;
	if (elementsOnLastPage < B / 2) //If elements on last page are less then half size of page
		bottomPages -= 1;
	if (bottomPages == 0) //Only root page
	{
		if (globalId == 0)
			sizeArray[currentNode] = elementNum;
		int id = globalId;
		while (id < elementNum)
		{
			keysArray[id] = hashesArray[id];
			id += threadsNum;
		}
		id = globalId;
		while (id < elementNum)
		{
			indexesArray[id] = valueArray[id];
			id += threadsNum;
		}
		if (globalId == 0)
		{
			indexesArray[B] = -1;
		}
		currentNode += 1;
	}
	else //Not only root page
	{
		//Creation of leafs
		int id = globalId;
		//Copying elements to leaf pages
		while (id < elementNum)
		{
			const int skippedPages = id / (B / 2); //Pages to skipped
			const int offsetOnPage = id - skippedPages * (B / 2); //Offset on page
			const int destination = skippedPages * B + offsetOnPage + (skippedPages == bottomPages ? -B : 0); /*Final destination where element must be copied.
			If a number of skipped pages equals to a number of all leaf pages then destination is corrected by size of pages to insert elements to last page.*/
			const int valuesDestination = skippedPages * (B + 1) + offsetOnPage + (skippedPages == bottomPages ? -(B + 1) : 0);
			keysArray[destination] = hashesArray[id];
			indexesArray[valuesDestination] = valueArray[id];
			id += threadsNum;
		}
		id = globalId;
		//Filling size of pages and indexes to next leafs
		while (id < bottomPages)
		{
			const int leftElements = elementNum - id * (B / 2);
			sizeArray[id] = B / 2 + (leftElements < (B / 2) ? leftElements : 0);
			indexesArray[id * (B + 1) + B] = id != (bottomPages - 1) ? id + 1 : -1;
			id += threadsNum;
		}
	}
	//Filling output
	if (globalId == 0)
	{
		reinterpret_cast<output_create_leafs*>(output)->rootNodeIndex = 0;
		reinterpret_cast<output_create_leafs*>(output)->usedNodes = bottomPages;
		reinterpret_cast<output_create_leafs*>(output)->isOnlyRoot = bottomPages == 0 ? 1 : 0;
	}
}

template<class HASH, int B, class Output>
__global__ void kernel_get_value(const int threadsNum, const int elementNum, HASH* keysArray, int* sizeArray, int* indexesArray, HASH* toFind, int height, int rootIndex, Output* output)
{
	const int globalId = GetGlobalId();
	const int sizeOfSizeNode = 1;
	const int sizeOfKeyNode = B;
	const int sizeOfIndexNode = B + 1;
	int currentHeight = 0;
	int currentNode = rootIndex;
	int i;
	int id = globalId;
	while (id < elementNum)
	{
		const HASH key = toFind[id];
		//Inner nodes
		while (currentHeight < height)
		{
			const int size = sizeArray[currentNode * sizeOfSizeNode];
			i = 0;
			while (i < size && keysArray[currentNode * sizeOfKeyNode + i] <= key)
				++i;
			currentNode = indexesArray[currentNode * sizeOfIndexNode + i];
			++currentHeight;
		}
		//Leaf level
		i = 0;
		const int size = sizeArray[currentNode * sizeOfSizeNode];
		int found = 0;
		while (i < size && keysArray[currentNode * sizeOfKeyNode + i] <= key)
		{
			if (key == keysArray[currentNode * sizeOfKeyNode + i])
			{
				if (std::is_same<Output, bool>::value)
					reinterpret_cast<bool*>(output)[id] = true;
				else
					output[id] = indexesArray[currentNode * sizeOfIndexNode + i];
				found = 1;
				break;
			}
			++i;
		}
		if (found == 0)
		{
			if (std::is_same<Output, bool>::value)
				reinterpret_cast<bool*>(output)[id] = false;
			else
				output[id] = -1;
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

	void insert(HASH key, int value) override;

	void bulk_insert(HASH* keys, int* values, int size) override;
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
	gpuErrchk(cudaMalloc(&d_hashes, size * sizeof(HASH)));
	gpuErrchk(cudaMalloc(&d_values, size* sizeof(int)));
	gpuErrchk(cudaMalloc(&d_output, sizeof(output_create_leafs)));

	gpuErrchk(cudaMemcpy(d_hashes, hashes, sizeof(HASH) * size, cudaMemcpyHostToDevice)); //Keys are copied to d_hashes
	gpuErrchk(cudaMemcpy(d_values, values, sizeof(int) * size, cudaMemcpyHostToDevice)); //Values are copied to d_values

	int threadsNum = elementNum < 1024 ? elementNum : 1024;
	int blocksNum = elementNum < 1024 ? 1 : static_cast<int>(std::ceil(elementNum / 1024.f));
	kernel_create_leafs<HASH, B> kernel_init(threadsNum, blocksNum) (threadsNum, elementNum, d_hashes, d_values, keysArray,
	                                                                 sizeArray, indexesArray, d_output);
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
				threadsNum, beginIndex, endIndex, indexesArray, keysArray, sizeArray, d_output);
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
}

template <class HASH, int B>
bool bplus_tree_gpu<HASH, B>::exist(HASH key)
{
	return exist(&key, 1)[0];
}

template <class HASH, int B>
std::vector<bool> bplus_tree_gpu<HASH, B>::exist(HASH* keys, int size)
{
	HASH* d_keys;
	bool *output = new bool[size];
	bool* d_output;
	gpuErrchk(cudaMalloc(&d_keys, size * sizeof(HASH)));
	gpuErrchk(cudaMalloc(&d_output, size * sizeof(bool)));
	gpuErrchk(cudaMemcpy(d_keys, keys, size * sizeof(HASH), cudaMemcpyHostToDevice));

	const int threadsNum = 32;
	kernel_get_value<HASH, B> kernel_init(threadsNum, 1) (threadsNum, size, keysArray, sizeArray, indexesArray, d_keys, height, rootNodeIndex, d_output);
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
	HASH* d_keys;
	std::vector<int> output(size);
	int* d_output;
	gpuErrchk(cudaMalloc(&d_keys, size * sizeof(HASH)));
	gpuErrchk(cudaMalloc(&d_output, size * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_keys, keys, size * sizeof(HASH), cudaMemcpyHostToDevice));

	const int threadsNum = 32;
	kernel_get_value<HASH, B> kernel_init(threadsNum, 1) (threadsNum, size, keysArray, sizeArray, indexesArray, d_keys, height, rootNodeIndex, d_output);
	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaMemcpy(output.data(), d_output, size * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_output));
	gpuErrchk(cudaFree(d_keys));
	return output;
}

template <class HASH, int B>
void bplus_tree_gpu<HASH, B>::insert(HASH key, int value)
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
