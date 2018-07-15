#pragma once
#include <vector>

template<class HASH, int B>
class bplus_tree
{
	static_assert((B % 2) == 0, "Size of page must be even.");
protected:
	static int needed_nodes(int elemNum);

	void virtual create_tree(HASH* keys, int* values, int size, char* suffixes, int suffixesLength) = 0;
public:
	virtual ~bplus_tree() = default;

	bool virtual exist(HASH key) = 0;
	std::vector<bool> virtual exist(HASH* keys, int size) = 0;

	/**
	 * \brief Checks if tree contains provided word.
	 * \param word Word to search. Reads until null byte.
	 * \return Value indicating if the provided word exists in the tree.
	 */
	bool virtual exist_word(char *word) = 0;
	/**
	 * \brief Checks if tree contains proivded words.
	 * \param words Words to search. An continuous array, where each word is ended with null byte.
	 * \param beginIndexes Indexes of begins of words from the 'words' array.
	 * \param size Size of the 'beginIndexes' array.
	 * \return Vector of values indicating which of provided words exists in the tree.
	 */
	std::vector<bool> virtual exist_word(char *words, int *beginIndexes, int size) = 0;

	int virtual get_value(HASH key) = 0;
	std::vector<int> virtual get_value(HASH* keys, int size) = 0;

	bool virtual insert(HASH key, int value) = 0;

	void virtual bulk_insert(HASH* keys, int* values, int size) = 0;

	int virtual get_height() = 0;
};

template <class HASH, int B>
int bplus_tree<HASH, B>::needed_nodes(int elemNum)
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
