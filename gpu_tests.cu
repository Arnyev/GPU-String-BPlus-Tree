#pragma once
#include "functions.h"
#include <algorithm>
#include "gpu_helper.cuh"
#include <thrust/execution_policy.h>

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

__device__ __host__ __inline__ ullong get_hash(const char* words, const int my_position)
{
	const int chars_to_hash = 13;

	ullong hash = 0;
	unsigned char last_bit = 1;
	unsigned char char_mask = CHARMASK;

	int i = 0;
	for (; i < chars_to_hash; i++)
	{
		const unsigned char c = words[i + my_position];
		if (c == BREAKCHAR)
		{
			char_mask = 0;
			last_bit = 0;
			break;
		}
		hash *= ALPHABETSIZE;
		hash += c & char_mask;
	}
	for (; i < chars_to_hash; i++)
	{
		hash *= ALPHABETSIZE;
	}
	if (!char_mask || words[chars_to_hash + my_position] == BREAKCHAR)
		last_bit = 0;

	return hash << 1 | last_bit;
}

__global__ void kernel_create_next_layer(const int threadsNum, const int beginIndex, const int endIndex, int* indexArray, ullong* keysArray, int* sizeArray, ullong *minArray)
{
	only_gpu_assert();
	const int globalId = GetGlobalId();
	const int minIndexesPerNode = TREEPAGE / 2 + 1;
	const int maxIndexesPerNode = TREEPAGE + 1;
	const int minKeysPerNode = TREEPAGE / 2;
	const int maxKeysPerNode = TREEPAGE;
	const int createdNodes = endIndex - beginIndex; //How many nodes were last time created
													//Creation of new layer
	int toCreate = my_max(1, createdNodes / (TREEPAGE / 2 + 1)); //How many nodes will be created in this iteration
	if (toCreate <= 0)
		toCreate = 1;
	//In each node there will be at least TREEPAGE / 2 keys and TREEPAGE / 2 + 1 indexes to lower layer nodes
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

__global__ void kernel_create_leafs(const int threads_num, const int element_num, const ullong* hashes_array,
	const int* value_array, ullong* keys_array, int* size_array, int* indexes_array, ullong* min_array)
{
	only_gpu_assert();
	const int globalId = GetGlobalId();
	const int maxIndexesPerNode = TREEPAGE + 1;
	const int minKeysPerNode = TREEPAGE / 2;
	const int maxKeysPerNode = TREEPAGE;
	int bottomPages = my_max(1, element_num * 2 / TREEPAGE); //How many pages will be created
	const int elementsOnLastPage = element_num - (bottomPages - 1) * TREEPAGE / 2;
	if (elementsOnLastPage < TREEPAGE / 2 && bottomPages > 1) //If elements on last page are less then half size of page
		bottomPages -= 1;
	int id = globalId;
	while (id < element_num)
	{
		int _nodeIndex = id / minKeysPerNode;
		const bool addToPrevious = _nodeIndex == bottomPages;
		_nodeIndex += addToPrevious ? -1 : 0;
		const int& nodeIndex = _nodeIndex;
		const int indexInNode = id - nodeIndex * minKeysPerNode;
		keys_array[nodeIndex * maxKeysPerNode + indexInNode] = hashes_array[id];
		indexes_array[nodeIndex * maxIndexesPerNode + indexInNode] = value_array[id];
		id += threads_num;
	}
	id = globalId;
	while (id < bottomPages)
	{
		const int &node_index = id;
		const bool is_last = node_index == bottomPages - 1;
		size_array[node_index] = is_last ? element_num - (bottomPages - 1) * minKeysPerNode : minKeysPerNode;
		min_array[node_index] = hashes_array[node_index * minKeysPerNode];
		indexes_array[node_index * maxIndexesPerNode + maxIndexesPerNode - 1] = is_last ? -1 : node_index + 1;
		id += threads_num;
	}
}

__global__ void kernel_find_words_v2(const int threadsNum, const ullong* keys_array, const int* indexes_array, const int* size_array,
	const int root_index, const int height, const char* suffixes, const int suffixes_size, const int elements_num, const char* words,
	const int* begin_indexes, bool* output)
{
	const int globalId = GetGlobalId();
	const int maxIndexesPerNode = TREEPAGE + 1;
	const int maxKeysPerNode = TREEPAGE;
	int id = globalId;
	while (id < elements_num)
	{
		const int beginIdx = begin_indexes[id];
		const ullong key = get_hash(words, beginIdx);
		int currentHeight = 0;
		int node = root_index;
		//Inner nodes
		while (currentHeight < height)
		{
			const int size = size_array[node];
			const ullong *keys_begin = keys_array + node * maxKeysPerNode;
			const ullong *keys_end = keys_begin + size;

			while (keys_begin + 1 != keys_end)
			{
				const ullong* keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			if (*keys_begin <= key)
				++keys_begin;
			node = indexes_array[node * maxIndexesPerNode + keys_begin - (keys_array + node * maxKeysPerNode)];
			//node = indexesArray[node * maxIndexesPerNode + (keys - keys_begin)];
			currentHeight += 1;
		}
		int suffixIdx, endSuffixIdx = -1;
		//Leaf level
		{
			const int size = size_array[node];
			const ullong *keys_begin = keys_array + node * maxKeysPerNode;
			const ullong *keys_end = keys_begin + size;
			const ullong *keys;
			while (keys_begin + 1 != keys_end)
			{
				keys = keys_begin + ((keys_end - keys_begin) >> 1);
				if (*keys <= key)
					keys_begin = keys;
				else
					keys_end = keys;
			}
			keys = keys_begin;
			keys_begin = keys_array + node * maxKeysPerNode;
			keys_end = keys_begin + size;
			if (keys < keys_end && *keys == key)
			{
				const int indexInKeyArray = keys - keys_begin;
				suffixIdx = indexes_array[node * maxIndexesPerNode + indexInKeyArray];
				if (indexInKeyArray < size - 1) //Next element is in the same leaf
				{
					endSuffixIdx = indexes_array[node * maxIndexesPerNode + indexInKeyArray + 1];
				}
				else //Next element is in the next leaf
				{
					if (indexes_array[node * maxIndexesPerNode + maxIndexesPerNode - 1] != -1) //Next leaf exists
					{
						endSuffixIdx = indexes_array[(node + 1) * maxIndexesPerNode];
					}
					else //It is the last element in the last leaf
					{
						endSuffixIdx = suffixes_size;
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
			const char nullTREEPAGEyte = static_cast<char>(0);
			const char *endSuffixIt = suffixes + endSuffixIdx;
			for (const char *suffixIt = suffixes + suffixIdx; suffixIt < endSuffixIt; ++suffixIt)
			{
				const char *wordIt = words + beginIdx + CHARSTOHASH; //Pointer to suffix of the word
				while (*suffixIt != nullTREEPAGEyte && *wordIt != nullTREEPAGEyte)
				{
					if (*suffixIt != *wordIt)
						break;
					++suffixIt;
					++wordIt;
				}
				if (*suffixIt == nullTREEPAGEyte && *wordIt == nullTREEPAGEyte)
				{
					result = true;
					break;
				}
				while (*suffixIt != nullTREEPAGEyte)
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

using namespace thrust;

struct bplus_tree_gpu_static
{
	device_vector<char> suffixes;
	device_vector<int> indexes;
	device_vector<ullong> keys;
	device_vector<int> sizes;
	device_vector<ullong> mins;
	int usedNodes;
	int rootNodeIndex;
	int height;
};

int needed_nodes(int elem_num)
{
	if (elem_num < TREEPAGE)
		return 1;
	int pages = elem_num * 2 / TREEPAGE;
	elem_num = pages;
	while (elem_num > TREEPAGE + 1)
	{
		elem_num = elem_num / (TREEPAGE / 2 + 1);
		pages += elem_num;
	}
	pages += 1;
	return pages;
}

void create_tree(bplus_tree_gpu_static & tree_gpu, const device_vector<ullong>& hashes, const device_vector<int>& values, const device_vector<uchar>& suffixes)
{
	const auto size = static_cast<int>(hashes.size());
	tree_gpu.height = 0;
	const auto node_count = needed_nodes(size);

	tree_gpu.suffixes.resize(suffixes.size());
	thrust::copy(thrust::device,suffixes.begin(), suffixes.end(), tree_gpu.suffixes.begin());
	tree_gpu.indexes.resize(node_count*(TREEPAGE + 1));
	tree_gpu.keys.resize(node_count*TREEPAGE);
	tree_gpu.sizes.resize(node_count);
	tree_gpu.mins.resize(node_count);

	int blocks_num = size <= 32 ? 1 : 2;
	int threads_num = size <= 32 ? 32 : std::min(size / 2, 1024);

	kernel_create_leafs kernel_init(blocks_num, threads_num) (threads_num, size, hashes.data().get(), values.data().get(),
		tree_gpu.keys.data().get(), tree_gpu.sizes.data().get(), tree_gpu.indexes.data().get(), tree_gpu.mins.data().get());

	gpuErrchk(cudaGetLastError());

	int last_created = std::max(1, size * 2 / TREEPAGE);
	int begin_index = 0;
	int end_index = last_created;
	while (last_created != 1)
	{
		tree_gpu.height += 1;

		blocks_num = last_created <= 32 ? 1 : 2;
		threads_num = last_created <= 32 ? 32 : std::min(last_created / 2, 1024);

		kernel_create_next_layer kernel_init(blocks_num, threads_num) (threads_num, begin_index, end_index,
			tree_gpu.indexes.data().get(), tree_gpu.keys.data().get(), tree_gpu.sizes.data().get(), tree_gpu.mins.data().get());

		gpuErrchk(cudaGetLastError());

		last_created = std::max(1, last_created / (TREEPAGE / 2 + 1));
		begin_index = end_index;
		end_index = end_index + last_created;
	}

	tree_gpu.rootNodeIndex = end_index - 1;
	tree_gpu.usedNodes = end_index;
}

void exist_word(const bplus_tree_gpu_static & tree_gpu, const device_vector<char>& words, const device_vector<int>& indexes, device_vector<bool>& output)
{
	const int count = static_cast<int>(indexes.size());
	output.resize(count);

	const int blocksNum = count <= 32 ? 1 : 2;
	const int threadsNum = count <= 32 ? 32 : std::min(count / 2, 1024);

	kernel_find_words_v2 kernel_init(blocksNum, threadsNum)(threadsNum, tree_gpu.keys.data().get(), tree_gpu.indexes.data().get(),
		tree_gpu.sizes.data().get(), tree_gpu.rootNodeIndex, tree_gpu.height, tree_gpu.suffixes.data().get(), tree_gpu.suffixes.size(),
		count, words.data().get(), indexes.data().get(), output.data().get());

	gpuErrchk(cudaGetLastError());
}

void prepare_for_search_tree(const host_vector<int>& positions_book_host, const host_vector<uchar>& words_book_host,
	const host_vector<int>& positions_dictionary_host, const host_vector<uchar>& words_dictionary_host,
	bplus_tree_gpu_static& tree, device_vector<int>& positions_book, device_vector<char>& words_book)
{
	device_vector<int> positions_dictionary(positions_dictionary_host);
	const device_vector<uchar> words_dictionary(words_dictionary_host);
	sorting_output_gpu output;

	create_output(words_dictionary, positions_dictionary, output);

	create_tree(tree, output.hashes, output.positions, output.suffixes);

	positions_book.resize(positions_book_host.size());
	copy(positions_book_host.begin(), positions_book_host.end(), positions_book.begin());

	words_book.resize(words_book_host.size());
	copy(words_book_host.begin(), words_book_host.end(), words_book.begin());
}

void test_gpu_tree_vectors(const char* dictionary_filename, const char* book_filename)
{
	host_vector<int> positions_book_host;
	host_vector<uchar> words_book_host;
	read_file(book_filename, positions_book_host, words_book_host);

	host_vector<int> positions_dictionary_host;
	host_vector<uchar> words_dictionary_host;
	read_file(dictionary_filename, positions_dictionary_host, words_dictionary_host);

	bplus_tree_gpu_static tree;
	device_vector<int> positions_book;
	device_vector<char> words_book;
	const auto build_time = measure::execution_gpu(prepare_for_search_tree, positions_book_host, words_book_host,
		positions_dictionary_host, words_dictionary_host, tree, positions_book, words_book);

	device_vector<bool> gpu_result;
	const auto execution_time = measure::execution_gpu(exist_word, tree, words_book, positions_book, gpu_result);

	std::vector<std::string> strings;
	create_strings(words_book_host, positions_book_host, strings);

	std::vector<bool> result(gpu_result.size());
	thrust::copy(gpu_result.begin(), gpu_result.end(), result.begin());

	std::vector<bool> cpu_result;
	get_cpu_result(words_dictionary_host, words_book_host, positions_book_host, cpu_result);

	int true_count = 0;

	for (int i = 0; i < result.size(); i++)
	{
		if (cpu_result[i] != result[i])
			std::cout << strings[i] << std::endl;

		if (result[i])
			true_count++;
	}

	append_to_csv("B - Plus Tree search", build_time / 1000, execution_time / 1000, positions_dictionary_host.size(),
		result.size(), static_cast<double>(true_count) / result.size());
}
