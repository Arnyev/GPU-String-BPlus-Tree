#pragma once

#if _DEBUG

#define CLEAR_MEMORY true

#define DEFAULT_MEMORY_VALUE (-1)

#endif

#include "bplus_tree.h"
#include "bplus_tree_gpu.cuh"
#include "gpu_helper.cuh"
#include <algorithm>
#include <iterator>
#include "not_implemented.h"
#include <array>


template <class HASH, int B>
class bplus_tree_cpu : public bplus_tree<HASH, B> 
{
	const int EXTEND_SIZE = 16;
	using index_array = std::array<int, B + 1>;
	using key_array = std::array<HASH, B>;
	std::vector<index_array> indexesArray;
	std::vector<key_array> keysArray;
	std::vector<int> sizeArray;
	std::vector<int> minArray;
	int reservedNodes;
	int usedNodes;
	int rootNodeIndex;
	int height;
	

	/**
	 * \brief Inserts a new element to a node. Splits if necessery and returns in reference parameters values of newly craeted node.
	 * \tparam IsLeafNode Is insertion happening on leaf node
	 * \param key Key to insert. After insertion, key will be filled with first key from newly created node.
	 * \param value Value to insert. After insertion, value will be filled with index of newly created node.
	 * \param node Node to insert and split.
	 * \param target Index where to insert a new element.
	 * \return Is new node where created?
	 */
	template<bool IsLeafNode>
	bool insert_element_at(HASH& key, int& value, int node, int target);

	/**
	 * \brief Inserts a new element to a node and then splits it in half. Reference parameters will be updated with values of newly created node.
	 * \tparam IsLeafNode Is insertion happening on leaf node
	 * \param key Key to insert. After insertion, key will be filled with first key from newly created node.
	 * \param value Value to insert. After insertion, value will be filled with index of newly created node.
	 * \param node Node to insert and split.
	 * \param target Index where to insert a new element.
	 */
	template<bool IsLeafNode>
	void insert_and_split(HASH& key, int& value, int node, int target);

	/**
	 * \brief Creates a new node.
	 * \return Index of a new node.
	 */
	int create_new_node();

	/**
	 * \brief Inserts key and value to node at given index by shifting elements to right
	 * \tparam IsLeafNode Is insertion happening on leaf node.
	 * \param key Key to insert.
	 * \param value Value to insert.
	 * \param node Index of a node.
	 * \param target Index where to insert.
	 */
	template <bool IsLeafNode>
	void insert_to_node_after(HASH key, int value, int node, int target);

	/**
	 * \brief Splits node from one level lower. The new created node contains greater elements.
	 * \tparam IsLeafNode Is insertion happening on leaf node.
	 * \param nodeToSplit Index of a node to split.
	 * \return Index of new created node.
	 */
	template <bool IsLeafNode>
	int split_node(int nodeToSplit);

	/**
	 * \brief 
	 * \param key Key to insert.
	 * \param value Value to insert.
	 * \param node Index of next node.
	 * \param height Height of next node.
	 * \param success Is element was inserted?
	 * \return Is node full?
	 */
	bool inner_insert(HASH& key, int& value, int node, int height, bool &success);
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

	bool insert(HASH key, int value) override;

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
bool bplus_tree_cpu<HASH, B>::insert(HASH key, int value)
{
	bool success;
	if (inner_insert(key, value, rootNodeIndex, 0, success))
	{
		//Need to add new node, increasing tree height
		const int oldNode = rootNodeIndex;
		const int newNode = value;
		rootNodeIndex = create_new_node();
		keysArray[rootNodeIndex][0] = minArray[newNode];
		minArray[rootNodeIndex] = minArray[oldNode];
		indexesArray[rootNodeIndex][0] = oldNode;
		indexesArray[rootNodeIndex][1] = newNode;
		sizeArray[rootNodeIndex] = 1;
		height += 1;
	}
	return success;
}

template <class HASH, int B>
void bplus_tree_cpu<HASH, B>::bulk_insert(HASH* keys, int* values, int size)
{
	throw not_implemented();
}

template <class HASH, int B>
template <bool IsLeafNode>
bool bplus_tree_cpu<HASH, B>::insert_element_at(HASH& key, int& value, int node, int target)
{
	const int size = sizeArray[node];
	if (size == B)
	{
		//Node need to be splited
		insert_and_split<IsLeafNode>(key, value, node, target - 1);
		//Upper node must insert a new element
		return true;
	}
	else
	{
		//Inserting new element to a node
		insert_to_node_after<IsLeafNode>(key, value, node, target - 1);
		//No new node for upper level
		return false;
	}
}

template <class HASH, int B>
template <bool IsLeafNode>
void bplus_tree_cpu<HASH, B>::insert_and_split(HASH& key, int& value, int node, int after)
{
	const int destination = after + 1;
	int newNode = create_new_node();
	// Right side of a new node
	const int newNode_right_offset = std::max(0, destination - B / 2);
	std::copy(keysArray[node].begin() + B / 2 + newNode_right_offset, keysArray[node].end(), keysArray[newNode].begin() + newNode_right_offset);
	if (IsLeafNode)
	{
		std::copy(indexesArray[node].begin() + B / 2 + newNode_right_offset, indexesArray[node].end() - 1, indexesArray[newNode].begin() + newNode_right_offset);
	}
	else
	{
		std::copy(indexesArray[node].begin() + B / 2 + newNode_right_offset + 1, indexesArray[node].end(), indexesArray[newNode].begin() + newNode_right_offset + 1);
	}
	// Left side of a new node
	const int newNode_left_offset = (B / 2 - newNode_right_offset) + (destination > B / 2 ? 1 : 0);
	std::copy(keysArray[node].begin() + B / 2, keysArray[node].end() - newNode_left_offset, keysArray[newNode].begin());
	if (IsLeafNode)
	{
		std::copy(indexesArray[node].begin() + B / 2, indexesArray[node].end() - newNode_left_offset - 1, indexesArray[newNode].begin());
	}
	else
	{
		std::copy(indexesArray[node].begin() + B / 2, indexesArray[node].end() - newNode_left_offset, indexesArray[newNode].begin());
	}
	if (destination < B / 2) // If element will not be inserted to a new node, shifting is required in an old node
	{
		// Right side of an old node
		std::rotate(keysArray[node].rbegin() + B / 2 - 1, keysArray[node].rbegin() + B / 2, keysArray[node].rend() - destination);
		if (IsLeafNode)
		{
			std::rotate(indexesArray[node].rbegin() + B / 2, indexesArray[node].rbegin() + B / 2 + 1, indexesArray[node].rend() - destination);
		}
		else
		{
			std::rotate(indexesArray[node].rbegin() + B / 2 - 1, indexesArray[node].rbegin() + B / 2, indexesArray[node].rend() - destination - 1);
		}
		// Left side of an old node is already in correct state
	}
	// Inserting a new element
	if (destination > B / 2) 
	{
		// Inserting to a new node
		keysArray[newNode][destination - B / 2 - 1] = key;
		if (IsLeafNode)
		{
			indexesArray[newNode][destination - B / 2 - 1] = value;
		}
		else
		{
			indexesArray[newNode][destination - B / 2] = value;
		}
	}
	else
	{
		// Inserting to an old node
		if (IsLeafNode)
		{
			keysArray[node][destination] = key;
			indexesArray[node][destination] = value;
		}
		else
		{
			indexesArray[node][destination + 1] = value;
			if (destination != B / 2)
			{
				keysArray[node][destination] = key;
			}
		}
	}
	sizeArray[newNode] = B / 2;
	sizeArray[node] = B / 2;
	if (IsLeafNode)
	{
		sizeArray[node] += 1;
		indexesArray[newNode][B] = indexesArray[node][B];
		indexesArray[node][B] = newNode;
		minArray[node] = keysArray[node][0];
		minArray[newNode] = keysArray[newNode][0];
	}
	else
	{
		minArray[node] = minArray[indexesArray[node][0]];
		minArray[newNode] = minArray[indexesArray[newNode][0]];
	}
	key = keysArray[newNode][0];
	value = newNode;
}

template <class HASH, int B>
int bplus_tree_cpu<HASH, B>::create_new_node()
{
	if (usedNodes == reservedNodes)
	{
		//No more space for nodes
		//New memory is allocated
		reservedNodes += EXTEND_SIZE;
		indexesArray.resize(reservedNodes);
		keysArray.resize(reservedNodes);
		sizeArray.resize(reservedNodes);
		minArray.resize(reservedNodes);
	}
	int result = usedNodes;
	usedNodes += 1;
	return result;
}

template <class HASH, int B>
template <bool IsLeafNode>
void bplus_tree_cpu<HASH, B>::insert_to_node_after(HASH key, int value, int node, int after)
{
	const int destination = after + 1;
	const int size = sizeArray[node];
	if (size == B)
		throw std::logic_error("Cannot insert. Page is full.");
	//Shifting elements one position to right
	if (!IsLeafNode)
	{
		//Saving most right index in inner node
		indexesArray[node][size + 1] = indexesArray[node][size];
	}
	for (int j = size; j > destination; --j)
	{
		keysArray[node][j] = keysArray[node][j - 1];
		indexesArray[node][j] = indexesArray[node][j - 1];
	}
	//Inserting new key and index
	keysArray[node][destination] = key;
	if (IsLeafNode)
	{
		indexesArray[node][destination] = value;
	}
	else
	{
		indexesArray[node][destination + 1] = value;
	}
	sizeArray[node] += 1;
	minArray[node] = keysArray[node][0];
}

template <class HASH, int B>
template <bool IsLeafNode>
int bplus_tree_cpu<HASH, B>::split_node(int nodeToSplit)
{
	const int newNode = create_new_node(); //New created node
	//Copying elements to a new node
	for (int j = 0; j < B / 2; ++j)
	{
		keysArray[newNode][j] = keysArray[nodeToSplit][B / 2 + j];
		indexesArray[newNode][j] = indexesArray[nodeToSplit][B / 2 + j];
	}
	if (IsLeafNode)
	{
		indexesArray[newNode][B] = indexesArray[nodeToSplit][B];
		indexesArray[nodeToSplit][B] = newNode;
	}
	else
	{
		indexesArray[newNode][B / 2] = indexesArray[nodeToSplit][B];
	}
#if _DEBUG
	for (int j = B / 2; j < B; ++j)
	{
		keysArray[newNode][j] = -1;
		indexesArray[newNode][j] = -1;
		keysArray[nodeToSplit][j] = -1;
		indexesArray[nodeToSplit][j] = -1;
	}
#endif

	//Setting size
	sizeArray[newNode] = B / 2;
	sizeArray[nodeToSplit] = B / 2;
	return newNode;
}

template <class HASH, int B>
bool bplus_tree_cpu<HASH, B>::inner_insert(HASH& key, int& value, int node, int height, bool &success)
{
	if (height == this->height)
	{
		//Leaf level
		int i = 0;
		const int size = sizeArray[node];
		while (i < size && keysArray[node][i] <= key)
		{
			if (key == keysArray[node][i])
			{
				//Key exists
				success = false;
				return false;
			}
			++i;
		}
		success = true;
		return insert_element_at<true>(key, value, node, i);
	}
	else
	{
		//Inner node level
		const int size = sizeArray[node];
		int i = 0;
		while (i < size && keysArray[node][i] <= key)
			++i;
		const int targetNode = indexesArray[node][i];
		const int target = i;
		if (!inner_insert(key, value, targetNode, height + 1, success))
		{
			//No more new elements
			return false;
		}
		else
		{
			//New element must be inserted
			//key - key of new element
			//value - value of new element and also index of newly created node
			return insert_element_at<false>(key, value, node, target);
		}
	}
}

template <class HASH, int B>
void bplus_tree_cpu<HASH, B>::create_tree(HASH* keys, int* values, int size)
{
	reservedNodes = needed_nodes(size);

	indexesArray = std::vector<index_array>(reservedNodes);
	keysArray = std::vector<key_array>(reservedNodes);
	sizeArray = std::vector<int>(reservedNodes);
	minArray = std::vector<int>(reservedNodes);
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
		std::copy(keys, values + size, keysArray[rootNodeIndex].begin());
		std::copy(values, values + size, indexesArray[rootNodeIndex].begin());
		minArray[rootNodeIndex] = keysArray[rootNodeIndex][0];
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
			std::copy(it, copyUntil, keysArray[currentNode].begin()); //Copying hashes 
			std::copy(itV, copyUntilV, indexesArray[currentNode].begin()); //Copying values
			sizeArray[currentNode] = static_cast<int>(std::distance(it, copyUntil));
			minArray[currentNode] = keysArray[currentNode][0];
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
				minArray[currentNode] = minArray[indexesArray[currentNode][0]];
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
			minArray[rootNodeIndex] = minArray[indexesArray[currentNode][0]];
			currentNode += 1;
		}
	}
	usedNodes = currentNode - 1;
}

template <class HASH, int B>
int bplus_tree_cpu<HASH, B>::get_leaf(HASH key)
{
	int currentHeight = 0;
	int node = rootNodeIndex;
	int i;
	//Inner nodes
	while (currentHeight < height)
	{
		const int size = sizeArray[node];
		i = 0;
		while (i < size && keysArray[node][i] <= key)
			++i;
		node = indexesArray[node][i];
		++currentHeight;
	}
	//Leaf level
	return node;
}

template <class HASH, int B>
bplus_tree_cpu<HASH, B>::bplus_tree_cpu(bplus_tree_gpu<HASH, B>& gpuTree)
{
	height = gpuTree.height;
	reservedNodes = gpuTree.reservedNodes;
	rootNodeIndex = gpuTree.rootNodeIndex;
	usedNodes = gpuTree.usedNodes;
	indexesArray = std::vector<index_array>(reservedNodes);
	keysArray = std::vector<key_array>(reservedNodes);
	sizeArray = std::vector<int>(reservedNodes);
	minArray = std::vector<int>(reservedNodes);
	gpuErrchk(cudaMemcpy(indexesArray.data(), gpuTree.indexesArray, reservedNodes * sizeof(HASH) * (B + 1),
		cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(keysArray.data(), gpuTree.keysArray, reservedNodes * sizeof(HASH) * B, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(sizeArray.data(), gpuTree.sizeArray, reservedNodes * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(minArray.data(), gpuTree.minArray, reservedNodes * sizeof(int), cudaMemcpyDeviceToHost));
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
	gpuErrchk(cudaMalloc(&gTree.minArray, reservedNodes * sizeof(int)));
	gpuErrchk(cudaMemcpy(gTree.indexesArray, indexesArray, reservedNodes * sizeof(HASH) * (B + 1),
		cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(gTree.keysArray, keysArray, reservedNodes * sizeof(HASH) * B, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(gTree.minArray, minArray, reservedNodes * sizeof(int), cudaMemcpyHostToDevice));
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
