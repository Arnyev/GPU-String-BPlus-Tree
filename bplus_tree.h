#pragma once

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
};
