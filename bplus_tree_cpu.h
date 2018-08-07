#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <array>

#include "bplus_tree.h"
#include "bplus_tree_gpu.cuh"
#include "gpu_helper.cuh"
#include "not_implemented.h"
#include "parameters.h"

template <class HASH, int B>
class bplus_tree_cpu : public bplus_tree<HASH, B> 
{
	const int EXTEND_SIZE = 16;
	using index_array = std::array<int, B + 1>;
	using key_array = std::array<HASH, B>;
	std::vector<char> suffixes;
	std::vector<index_array> indexesArray;
	std::vector<key_array> keysArray;
	std::vector<int> sizeArray;
	std::vector<HASH> minArray;
	int reservedNodes{};
	int usedNodes{};
	int rootNodeIndex{};
	int height{};

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
	 * \param after Index where to insert.
	 */
	template <bool IsLeafNode>
	void insert_to_node_after(HASH key, int value, int node, int after);

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
	void create_tree(HASH* keys, int* values, int size, const char* suffixes, int suffixesLength) override;

	int get_leaf(HASH key);
public:
	bplus_tree_cpu(bplus_tree_gpu<HASH, B>& gpu_tree);
	bplus_tree_gpu<HASH, B> export_to_gpu();

	bplus_tree_cpu(HASH* keys, int* values, int size, char* suffixes, int suffixesLength);

	bool exist(HASH key) override;
	std::vector<bool> exist(HASH* keys, int size) override;

	bool exist_word(const char *word) override;
	std::vector<bool> exist_word(const char *words, int wordsSize, int *beginIndexes, int indexSize) override;

	int get_value(HASH key) override;
	std::vector<int> get_value(HASH* keys, int size) override;

	bool insert(HASH key, int value) override;

	void bulk_insert(HASH* keys, int* values, int size) override;

	int get_height() override;
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
int bplus_tree_cpu<HASH, B>::get_height()
{
	return height;
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
	const int newNode = create_new_node();
	key_array& nodeKeys = keysArray[node];
	index_array& nodeIndexes = indexesArray[node];
	key_array& newNodeKeys = keysArray[newNode];
	index_array& newNodeIndexes = indexesArray[newNode];
	// Right side of a new node
	const int newNode_right_offset = std::max(0, destination - B / 2);
	std::copy(nodeKeys.begin() + B / 2 + newNode_right_offset, nodeKeys.end(), newNodeKeys.begin() + newNode_right_offset);
	if (IsLeafNode)
	{
		std::copy(nodeIndexes.begin() + B / 2 + newNode_right_offset, nodeIndexes.end() - 1, newNodeIndexes.begin() + newNode_right_offset);
	}
	else
	{
		std::copy(nodeIndexes.begin() + B / 2 + newNode_right_offset, nodeIndexes.end(), newNodeIndexes.begin() + newNode_right_offset);
	}
	if (destination > B / 2) // Element will be inserted to a new node, more copying is required
	{
		// Left side of a new node
		const int newNode_left_offset = B - destination;
		if (newNode_left_offset < B / 2)
		{
			std::copy(nodeKeys.begin() + B / 2 + 1, nodeKeys.end() - newNode_left_offset, newNodeKeys.begin());
		}
		if (IsLeafNode)
		{
			std::copy(nodeIndexes.begin() + B / 2 + 1, nodeIndexes.end() - newNode_left_offset - 1, newNodeIndexes.begin());
		}
		else
		{
			std::copy(nodeIndexes.begin() + B / 2 + 1, nodeIndexes.end() - newNode_left_offset, newNodeIndexes.begin());
		}
	}
	else //if (destination <= B / 2) // If element will not be inserted to a new node, shifting is required in an old node
	{
		// Right side of an old node
		std::rotate(nodeKeys.rbegin() + B / 2 - 1, nodeKeys.rbegin() + B / 2, nodeKeys.rend() - destination);
		if (IsLeafNode)
		{
			std::rotate(nodeIndexes.rbegin() + B / 2, nodeIndexes.rbegin() + B / 2 + 1, nodeIndexes.rend() - destination);
		}
		else
		{
			std::rotate(nodeIndexes.rbegin() + B / 2 - 1, nodeIndexes.rbegin() + B / 2, nodeIndexes.rend() - destination - 1);
		}
		// Left side of an old node is already in correct state
	}
	// Inserting a new element
	if (destination > B / 2) 
	{
		// Inserting to a new node
		newNodeKeys[destination - B / 2 - 1] = key;
		if (IsLeafNode)
		{
			newNodeIndexes[destination - B / 2 - 1] = value;
		}
		else
		{
			newNodeIndexes[destination - B / 2] = value;
		}
	}
	else
	{
		// Inserting to an old node
		if (IsLeafNode)
		{
			nodeKeys[destination] = key;
			nodeIndexes[destination] = value;
		}
		else
		{
			if (destination != B / 2)
			{
				nodeKeys[destination] = key;
				nodeIndexes[destination + 1] = value;
			}
			else
			{
				newNodeIndexes[0] = value;
			}
		}
	}
	sizeArray[newNode] = B / 2;
	sizeArray[node] = B / 2;
	if (IsLeafNode)
	{
		sizeArray[node] += 1;
		newNodeIndexes[B] = nodeIndexes[B];
		nodeIndexes[B] = newNode;
		minArray[node] = nodeKeys[0];
		minArray[newNode] = newNodeKeys[0];
	}
	else
	{
		minArray[node] = minArray[nodeIndexes[0]];
		minArray[newNode] = minArray[newNodeIndexes[0]];
	}
	key = minArray[newNode];
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
	key_array& nodeKeys = keysArray[node];
	index_array& nodeIndexes = indexesArray[node];
	if (size == B)
		throw std::logic_error("Cannot insert. Page is full.");
	//Shifting elements one position to right
	const int offset = B - size - 1;
	std::rotate(nodeKeys.rbegin() + offset, nodeKeys.rbegin() + offset + 1, nodeKeys.rend() - destination);
	if (IsLeafNode)
	{
		std::rotate(nodeIndexes.rbegin() + offset + 1, nodeIndexes.rbegin() + offset + 2, nodeIndexes.rend() - destination);
	}
	else
	{
		std::rotate(nodeIndexes.rbegin() + offset, nodeIndexes.rbegin() + offset + 1, nodeIndexes.rend() - destination - 1);
	}
	//Inserting new key and index
	nodeKeys[destination] = key;
	if (IsLeafNode)
	{
		nodeIndexes[destination] = value;
	}
	else
	{
		nodeIndexes[destination + 1] = value;
	}
	sizeArray[node] += 1;
	minArray[node] = nodeKeys[0];
}

template <class HASH, int B>
bool bplus_tree_cpu<HASH, B>::inner_insert(HASH& key, int& value, int node, int height, bool &success)
{
	const int size = sizeArray[node];
	const key_array& nodeKeys = keysArray[node];
	const index_array& nodeIndexes = indexesArray[node];
	const auto found = std::lower_bound(nodeKeys.begin(), nodeKeys.begin() + size, key);
	if (height == this->height)
	{
		//Leaf level
		success = found == nodeKeys.end() || *found != key;
		if (!success)
			return false;
		return insert_element_at<true>(key, value, node, static_cast<int>(std::distance(nodeKeys.begin(), found)));
	}
	else
	{
		//Inner node level
		const int target = static_cast<int>(std::distance(nodeKeys.begin(), found));
		const int targetNode = nodeIndexes[target];
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
void bplus_tree_cpu<HASH, B>::create_tree(HASH* keys, int* values, int size, const char* suffixes, int suffixesLength)
{
	reservedNodes = needed_nodes(size);
	this->suffixes = std::vector<char>(suffixes, suffixes + suffixesLength);
	indexesArray = std::vector<index_array>(reservedNodes);
	keysArray = std::vector<key_array>(reservedNodes);
	sizeArray = std::vector<int>(reservedNodes);
	minArray = std::vector<HASH>(reservedNodes);
	int node = 0; //Index of first not initilize node
	int bottomPages = std::max(1, size * 2 / B);
	const int elementsOnLastPage = size - (bottomPages - 1) * B / 2;
	if (elementsOnLastPage < B / 2 && bottomPages > 1) //If elements on last page are less then half size of page
		bottomPages -= 1;
	height = 0;
	HASH* it = keys;
	int* itV = values;
	//Creation of leafs
	for (int i = 0; i < bottomPages; ++i)
	{
		key_array& nodeKeys = keysArray[node];
		index_array& nodeIndexes = indexesArray[node];
		HASH* copyUntil;
		int* copyUntilV;
		if (i == bottomPages - 1) //Last page
		{
			copyUntil = keys + size;
			copyUntilV = values + size;
			nodeIndexes[B] = -1; //There is no next page
		}
		else
		{
			copyUntil = it + B / 2;
			copyUntilV = itV + B / 2;
			nodeIndexes[B] = node + 1; //Next page
		}
		std::copy(it, copyUntil, nodeKeys.begin()); //Copying keys 
		std::copy(itV, copyUntilV, nodeIndexes.begin()); //Copying values
		sizeArray[node] = static_cast<int>(std::distance(it, copyUntil));
		minArray[node] = nodeKeys[0];
		it += B / 2;
		itV += B / 2;
		node += 1;
	}
	int firstNode = 0; //First index of nodes from previous layer
	int lastNode = node; //Index after last used node
	int createdNodes = bottomPages; //How many nodes were created
	while (createdNodes > 1) //More than 1 nodes were created in the last iteration, next layer is required
	{
		//Creation of new layer
		height += 1; //Height of a tree is increasing
		int toCreate = std::max(1, createdNodes / (B / 2 + 1)); //How many nodes will be created
		//In each node there will be at least B / 2 keys and B / 2 + 1 indexes to lower layer nodes
		int thisNodeIndexesBegin = firstNode; //Begin of indexes which gonna be included in new node.
		int thisNodeIndexesEnd; //End of indexes which gonna be included in new node. (Last index + 1)
		for (int i = 0; i < toCreate; ++i)
		{
			key_array& nodeKeys = keysArray[node];
			index_array& nodeIndexes = indexesArray[node];
			if (i == toCreate - 1) //Last page
				thisNodeIndexesEnd = lastNode;
			else
				thisNodeIndexesEnd = thisNodeIndexesBegin + B / 2 + 1;
			const int distance = thisNodeIndexesEnd - thisNodeIndexesBegin;
			const int offset = thisNodeIndexesBegin + 1;
			std::iota(nodeIndexes.begin(), nodeIndexes.begin() + distance, thisNodeIndexesBegin);
			std::copy(minArray.begin() + offset, minArray.begin() + offset + distance - 1, nodeKeys.begin());
			sizeArray[node] = distance - 1;
			minArray[node] = minArray[nodeIndexes[0]];
			thisNodeIndexesBegin += B / 2 + 1;
			node += 1;
		}
		createdNodes = toCreate;
		firstNode = lastNode;
		lastNode = node;
	}
	rootNodeIndex = node - 1;
	usedNodes = node;
}

template <class HASH, int B>
int bplus_tree_cpu<HASH, B>::get_leaf(HASH key)
{
	int currentHeight = 0;
	int node = rootNodeIndex;
	//Inner nodes
	while (currentHeight < height)
	{
		key_array& nodeKeys = keysArray[node];
		index_array& nodeIndexes = indexesArray[node];
		const int size = sizeArray[node];
		const auto found = std::lower_bound(nodeKeys.begin(), nodeKeys.begin() + size, key);
		const int index = static_cast<int>(std::distance(nodeKeys.begin(), found) + (found != nodeKeys.end() && *found == key ? 1 : 0));
		node = nodeIndexes[index];
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
bplus_tree_cpu<HASH, B>::bplus_tree_cpu(HASH* keys, int* values, int size, char* suffixes, int suffixesLength)
{
	bplus_tree_cpu<HASH, B>::create_tree(keys, values, size, suffixes, suffixesLength);
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
bool bplus_tree_cpu<HASH, B>::exist_word(const char* word)
{
	const char nullByte = static_cast<char>(0);
	const int maxLen = 13;
	const int wordLen = static_cast<int>(strlen(word));
	const ullong hash = get_hash(reinterpret_cast<const uchar*>(word), CHARSTOHASH, 0);
	int index = -1;
	int endSuffixIndex = -1;
	{
		const int node = get_leaf(hash);
		//Leaf level
		key_array& nodeKeys = keysArray[node];
		index_array& nodeIndexes = indexesArray[node];
		const int size = sizeArray[node];
		const auto found = std::lower_bound(nodeKeys.begin(), nodeKeys.begin() + size, hash);
		if (found != nodeKeys.end() && *found == hash)
		{
			const int index2 = static_cast<int>(std::distance(nodeKeys.begin(), found));
			index = nodeIndexes[index2];
			if (index2 < size - 1) //Next element is in the same leaf
			{
				endSuffixIndex = nodeIndexes[index2 + 1];
			}
			else //Next element is in the next leaf
			{
				if (nodeIndexes[B] != -1) //Next leaf exists
				{
					endSuffixIndex = indexesArray[nodeIndexes[B]][0];
				}
				else //It is the last element in the last leaf
				{
					endSuffixIndex = static_cast<int>(suffixes.size());
				}
			}
		}
	}
	if (index < 0)
		return false;
	if (wordLen <= maxLen)
		return true;
	auto suffixesIt = suffixes.begin() + index;
	for (int suffixIndex = index; index < endSuffixIndex; ++index, ++suffixesIt)
	{
		const char *wordIt = word + maxLen;
		while (*suffixesIt != nullByte && *wordIt != nullByte)
		{
			if (*suffixesIt != *wordIt)
				break;
			++suffixesIt;
			++index;
			++wordIt;
		}
		if (*suffixesIt == nullByte && *wordIt == nullByte)
			return true;
		while (*suffixesIt != nullByte) {
			++suffixesIt;
			++index;
		}
	}
	return false;
}

template <class HASH, int B>
std::vector<bool> bplus_tree_cpu<HASH, B>::exist_word(const char* words, int wordsSize, int* beginIndexes, int indexSize)
{
	std::vector<bool> result(indexSize);
	for (int i = 0; i < indexSize; ++i)
	{
		result[i] = exist_word(words + beginIndexes[i]);
	}
	return result;
}

template <class HASH, int B>
int bplus_tree_cpu<HASH, B>::get_value(HASH key)
{
	const int node = get_leaf(key);
	//Leaf level
	key_array& nodeKeys = keysArray[node];
	index_array& nodeIndexes = indexesArray[node];
	const int size = sizeArray[node];
	const auto found = std::lower_bound(nodeKeys.begin(), nodeKeys.begin() + size, key);
	if (found != nodeKeys.end() && *found == key)
	{
		const int index = static_cast<int>(std::distance(nodeKeys.begin(), found));
		return nodeIndexes[index];
	}
	else
	{
		return -1;
	}
}
