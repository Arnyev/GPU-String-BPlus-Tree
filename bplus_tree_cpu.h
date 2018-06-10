#pragma once
#include "bplus_tree.h"
#include "bplus_tree_gpu.cuh"
#include "gpu_helper.cuh"
#include <algorithm>
#include <iterator>
#include "not_implemented.h"

template <class HASH, int B>
class bplus_tree_cpu : public bplus_tree<HASH, B> 
{
	using index_array = int[B + 1];
	using key_array = HASH[B];
	index_array* indexesArray;
	key_array* keysArray;
	int* sizeArray;
	int reservedNodes;
	int usedNodes;
	int rootNodeIndex;
	int height;
protected:
	void create_tree(HASH* keys, int* values, int size) override;

	int get_leaf(HASH key);
public:
	bplus_tree_cpu(bplus_tree_gpu<HASH, B>& gpu_tree);
	bplus_tree_gpu<HASH, B> export_to_gpu();

	bplus_tree_cpu(HASH* keys, int* values, int size);

	bool exist(HASH key) override;
	std::vector<bool> exist(HASH* keys, int size) override;

	int get_value(HASH key) override;
	std::vector<int> get_value(HASH* keys, int size) override;

	void insert(HASH key, int value) override;

	void bulk_insert(HASH* keys, int* values, int size) override;
};

template <class HASH, int B>
std::vector<int> bplus_tree_cpu<HASH, B>::get_value(HASH* keys, int size)
{
	std::vector<int> tmp(size);
	auto it = tmp.begin();
	for (int i = 0; i < size; ++i, ++it)
		*it = get_value(keys[i]);
	return tmp;
}

template <class HASH, int B>
void bplus_tree_cpu<HASH, B>::insert(HASH key, int value)
{
	throw not_implemented();
}

template <class HASH, int B>
void bplus_tree_cpu<HASH, B>::bulk_insert(HASH* keys, int* values, int size)
{
	throw not_implemented();
}

template <class HASH, int B>
void bplus_tree_cpu<HASH, B>::create_tree(HASH* keys, int* values, int size)
{
	reservedNodes = needed_nodes(size);

	indexesArray = new index_array[reservedNodes];
	keysArray = new key_array[reservedNodes];
	sizeArray = new int[reservedNodes];
	int currentNode = 0; //Index of first not initilize node
	int bottomPages = size * 2 / B;
	int elementsOnLastPage = size - (bottomPages - 1) * B / 2;
	if (elementsOnLastPage < B / 2) //If elements on last page are less then half size of page
		bottomPages -= 1;
	if (bottomPages == 0) //Only root page
	{
		height = 0;
		rootNodeIndex = currentNode;
		sizeArray[rootNodeIndex] = size;
		std::copy(keys, values + size, keysArray[rootNodeIndex]);
		std::copy(values, values + size, indexesArray[rootNodeIndex]);
		currentNode += 1;
	}
	else //Not only root page
	{
		height = 0;
		HASH* it = keys;
		int* itV = values;
		//Creation of leafs
		for (int i = 0; i < bottomPages; ++i)
		{
			HASH* copyUntil;
			int* copyUntilV;
			if (i == bottomPages - 1) //Last page
			{
				copyUntil = keys + size;
				copyUntilV = values + size;
				indexesArray[currentNode][B] = -1; //There is no next page
			}
			else
			{
				copyUntil = it + B / 2;
				copyUntilV = itV + B / 2;
				indexesArray[currentNode][B] = currentNode + 1; //Next page
			}
			std::copy(it, copyUntil, keysArray[currentNode]); //Copying hashes 
			std::copy(itV, copyUntilV, indexesArray[currentNode]); //Copying values
			sizeArray[currentNode] = std::distance(it, copyUntil);
			it += B / 2;
			itV += B / 2;
			currentNode += 1;
		}
		int firstNode = 0; //First index of nodes from previous layer
		int lastNode = currentNode; //Index after last used page
		int createdNodes = bottomPages; //How many nodes were created
		while (createdNodes > B) //If all indexes doesn't fit inside one node, new layer is required
		{
			//Creation of new layer
			height += 1; //Height of a tree is increasing
			int toCreate = createdNodes / (B / 2 + 1); //How many nodes will be created
			//In each node there will be at least B / 2 keys and B / 2 + 1 indexes to lower layer nodes
			int thisNodeIndexesBegin = firstNode; //Begin of indexes which gonna be included in new node.
			int thisNodeIndexesEnd; //End of indexes which gonna be included in new node. (Last index + 1)
			for (int i = 0; i < toCreate; ++i)
			{
				if (i == toCreate - 1) //Last page
					thisNodeIndexesEnd = lastNode;
				else
					thisNodeIndexesEnd = thisNodeIndexesBegin + B / 2 + 1;
				indexesArray[currentNode][0] = thisNodeIndexesBegin;
				for (int j = thisNodeIndexesBegin + 1, x = 0; j < thisNodeIndexesEnd; ++j, ++x)
				{
					keysArray[currentNode][x] = keysArray[j][0];
					indexesArray[currentNode][x + 1] = j;
				}
				sizeArray[currentNode] = thisNodeIndexesEnd - thisNodeIndexesBegin - 1;
				thisNodeIndexesBegin += B / 2 + 1;
				currentNode += 1;
			}
			createdNodes = toCreate;
			firstNode = lastNode;
			lastNode = currentNode;
		}
		//Root creation
		{
			height += 1;
			rootNodeIndex = currentNode;
			indexesArray[rootNodeIndex][0] = firstNode;
			for (int j = firstNode + 1, x = 0; j < lastNode; ++j, ++x)
			{
				keysArray[rootNodeIndex][x] = keysArray[j][0];
				indexesArray[rootNodeIndex][x + 1] = j;
			}
			sizeArray[rootNodeIndex] = lastNode - firstNode - 1;
			currentNode += 1;
		}
	}
	usedNodes = currentNode - 1;
}

template <class HASH, int B>
int bplus_tree_cpu<HASH, B>::get_leaf(HASH key)
{
	int currentHeight = 0;
	int currentNode = rootNodeIndex;
	int i;
	//Inner nodes
	while (currentHeight < height)
	{
		const int size = sizeArray[currentNode];
		i = 0;
		while (i < size && keysArray[currentNode][i] <= key)
			++i;
		currentNode = indexesArray[currentNode][i];
		++currentHeight;
	}
	//Leaf level
	return currentNode;
}

template <class HASH, int B>
bplus_tree_cpu<HASH, B>::bplus_tree_cpu(bplus_tree_gpu<HASH, B>& gpuTree)
{
	height = gpuTree.height;
	reservedNodes = gpuTree.reservedNodes;
	rootNodeIndex = gpuTree.rootNodeIndex;
	usedNodes = gpuTree.usedNodes;
	indexesArray = new index_array[reservedNodes];
	keysArray = new key_array[reservedNodes];
	sizeArray = new int[reservedNodes];
	gpuErrchk(cudaMemcpy(indexesArray, gpuTree.indexesArray, reservedNodes * sizeof(HASH) * (B + 1),
		cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(keysArray, gpuTree.keysArray, reservedNodes * sizeof(HASH) * B, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(sizeArray, gpuTree.sizeArray, reservedNodes * sizeof(int), cudaMemcpyDeviceToHost));
}

template <class HASH, int B>
bplus_tree_gpu<HASH, B> bplus_tree_cpu<HASH, B>::export_to_gpu()
{
	bplus_tree_gpu<HASH, B> gTree;
	gTree.height = height;
	gTree.reservedNodes = reservedNodes;
	gTree.rootNodeIndex = rootNodeIndex;
	gTree.usedNodes = usedNodes;
	gpuErrchk(cudaMalloc(&gTree.indexesArray, reservedNodes * sizeof(HASH) * (B + 1)));
	gpuErrchk(cudaMalloc(&gTree.keysArray, reservedNodes * sizeof(HASH) * B));
	gpuErrchk(cudaMalloc(&gTree.sizeArray, reservedNodes * sizeof(int)));
	gpuErrchk(cudaMemcpy(gTree.indexesArray, indexesArray, reservedNodes * sizeof(HASH) * (B + 1),
		cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(gTree.keysArray, keysArray, reservedNodes * sizeof(HASH) * B, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(gTree.sizeArray, sizeArray, reservedNodes * sizeof(int), cudaMemcpyHostToDevice));
	return gTree;
}

template <class HASH, int B>
bplus_tree_cpu<HASH, B>::bplus_tree_cpu(HASH* keys, int* values, int size)
{
	create_tree(keys, values, size);
}

template <class HASH, int B>
bool bplus_tree_cpu<HASH, B>::exist(HASH key)
{
	return get_value(key) >= 0;
}

template <class HASH, int B>
std::vector<bool> bplus_tree_cpu<HASH, B>::exist(HASH* keys, int size)
{
	auto values = get_value(keys, size);
	std::vector<bool> tmp(size);
	std::transform(values.begin(), values.end(), tmp.begin(), [](int i) -> bool {return i >= 0; });
	return tmp;
}

template <class HASH, int B>
int bplus_tree_cpu<HASH, B>::get_value(HASH key)
{
	const int currentNode = get_leaf(key);
	//Leaf level
	int i = 0;
	const int size = sizeArray[currentNode];
	while (i < size && keysArray[currentNode][i] <= key)
	{
		if (key == keysArray[currentNode][i])
			return indexesArray[currentNode][i];
		++i;
	}
	return -1;
}
