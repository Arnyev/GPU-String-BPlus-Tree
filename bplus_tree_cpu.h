#pragma once
#include "bplus_tree.h"
#include "bplus_tree_gpu.cuh"
#include "gpu_helper.cuh"
#include <algorithm>
#include <iterator>

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
public:
	bplus_tree_cpu(bplus_tree_gpu<HASH, B>& gpu_tree);
	bplus_tree_gpu<HASH, B> export_to_gpu();

	template <class IteratorKeys, class IteratorValues>
	bplus_tree_cpu(IteratorKeys firstKeys, IteratorKeys lastKeys, IteratorValues firstValues, IteratorValues lastValues);
};

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
template <class IteratorKeys, class IteratorValues>
bplus_tree_cpu<HASH, B>::bplus_tree_cpu(IteratorKeys firstKeys, IteratorKeys lastKeys, IteratorValues firstValues, IteratorValues lastValues)
{
	int elementNum = lastKeys - firstKeys; //Number of hashes
	//reservedNodes = std::max(static_cast<double>(1), log(elementNum) / log(B));
	reservedNodes = needed_nodes(elementNum);

	indexesArray = new index_array[reservedNodes];
	keysArray = new key_array[reservedNodes];
	sizeArray = new int[reservedNodes];
	int currentNode = 0; //Index of first not initilize node
	int bottomPages = elementNum * 2 / B;
	int elementsOnLastPage = elementNum - (bottomPages - 1) * B / 2;
	if (elementsOnLastPage < B / 2) //If elements on last page are less then half size of page
		bottomPages -= 1;
	if (bottomPages == 0) //Only root page
	{
		height = 0;
		rootNodeIndex = currentNode;
		sizeArray[rootNodeIndex] = elementNum;
		std::copy(firstKeys, lastKeys, keysArray[rootNodeIndex]);
		std::copy(firstValues, lastValues, indexesArray[rootNodeIndex]);
		currentNode += 1;
	}
	else //Not only root page
	{
		height = 1;
		IteratorKeys it = firstKeys;
		IteratorValues itV = firstValues;
		//Creation of leafs
		for (int i = 0; i < bottomPages; ++i)
		{
			IteratorKeys copyUntil;
			IteratorValues copyUntilV;
			if (i == bottomPages - 1) //Last page
			{
				copyUntil = lastKeys;
				copyUntilV = lastValues;
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
