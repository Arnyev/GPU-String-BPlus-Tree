#pragma once
#include <vector>

template<class HASH, int B>
class bplus_tree
{
private:
	int inner_needed(int elemNum)
	{
		if (elemNum < B)
			return 1;
		int fullPages = elemNum / (B / 2 + 1);
		return fullPages + inner_needed(fullPages);
	}
protected:
	int needed_nodes(int elemNum)
	{
		if (elemNum < B)
			return 1;
		int fullPages = elemNum * 2 / B;
		return fullPages + inner_needed(fullPages);
	}
public:
	virtual ~bplus_tree() = default;

	bool virtual exist(HASH key) = 0;
	std::vector<bool> virtual exist(HASH* keys, int size) = 0;

	int virtual get_value(HASH key) = 0;
	std::vector<int> virtual get_value(HASH* keys, int size) = 0;
};
