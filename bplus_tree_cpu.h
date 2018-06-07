#pragma once
#include "bplus_tree.h"
#include "bplus_tree_gpu.cuh"
#include "gpu_helper.cuh"

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
	bplus_tree_cpu(bplus_tree_gpu<HASH, B> gpu_tree)
	{
		height = gpu_tree.height;
		reservedNodes = gpu_tree.reservedNodes;
		rootNodeIndex = gpu_tree.rootNodeIndex;
		usedNodes = gpu_tree.usedNodes;
		indexesArray = new index_array[reservedNodes];
		keysArray = new key_array[reservedNodes];
		sizeArray = new int[reservedNodes];
		gpuErrchk(cudaMemcpy(indexesArray, gpu_tree.indexesArray, reservedNodes * sizeof(HASH) * (B + 1), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(keysArray, gpu_tree.keysArray, reservedNodes * sizeof(HASH) * B, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(sizeArray, gpu_tree.sizeArray, reservedNodes * sizeof(int), cudaMemcpyDeviceToHost));
	}

	template<class Iterator>
	bplus_tree_cpu(Iterator first, Iterator last)
	{
		int elementNum = last - first; //Number of hashes
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
			copy(first, last, keysArray[rootNodeIndex]);
			//TODO indexes should point to array of string
			currentNode += 1;
		}
		else //Not only root page
		{
			height = 1;
			Iterator it = first;
			//Creation of leafs
			for (int i = 0; i < bottomPages; ++i)
			{
				Iterator copyUntil;
				if (i == bottomPages - 1) //Last page
				{
					copyUntil = last;
					indexesArray[currentNode][B] = -1; //There is no next page
				}
				else
				{
					copyUntil = it + B / 2;
					indexesArray[currentNode][B] = currentNode + 1; //Next page
				}
				copy(it, copyUntil, keysArray[currentNode]); //Copying hashes 
				sizeArray[currentNode] = copyUntil - it;
				it += B / 2;
				currentNode += 1;
			}
			int firstNode = 0; //First index of nodes from previous layer
			int lastNode = currentNode; //Index after last used page
			int createdNodes = bottomPages; //How many nodes were created
			while (createdNodes > B) //If all indexes doesn't fit inside one node, new layer is required
			{	//Creation of new layer
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
};
