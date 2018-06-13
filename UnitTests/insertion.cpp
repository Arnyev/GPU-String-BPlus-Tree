#include "../bplus_tree_cpu.h"
#ifdef BOOST_ENABLE
#define BOOST_TEST_MODULE BPlusTreeTests
#include <boost/test/unit_test.hpp>

using HASH = int;
struct tree_pair
{
	HASH key;
	int value;
};

template<int PageSize>
bplus_tree_cpu<HASH, PageSize> initialize_tree(const int elements, std::vector<tree_pair> &pairs)
{
	std::vector<HASH> keys(elements);
	std::vector<int> values(elements);
	int k = 0;
	std::transform(keys.begin(), keys.end(), keys.begin(), [&k](HASH) -> HASH { return (++k) * 10; });
	k = elements + 1;
	std::transform(values.begin(), values.end(), values.begin(), [&k](HASH) -> HASH { return (--k) * 10; });
	for (int i = 0; i < keys.size(); ++i)
	{
		pairs.push_back(tree_pair{ keys[i], values[i] });
	}
	return bplus_tree_cpu<HASH, PageSize> (keys.data(), values.data(), elements);
}

template<int PageSize>
bool check_if_all_key_exists(bplus_tree_cpu<HASH, PageSize> &tree, std::vector<tree_pair> &pairs)
{
	for (auto& pair : pairs)
	{
		int expectedValue = pair.value;
		int returnedValue = tree.get_value(pair.key);
		if (expectedValue != returnedValue)
		{
			BOOST_CHECK_MESSAGE(false, "Wrong returned value. Tree[" << pair.key << "] returned " << returnedValue << " instead of " << expectedValue << ".");
			return false;
		}
	}
	return true;
}

BOOST_AUTO_TEST_CASE(cpu_insertion_to_root_without_node_spliting)
{
	const int PAGE_SIZE = 16;
	const int INITIAL_SIZE = 12;
	std::vector<tree_pair> pairs;
	auto tree = initialize_tree<PAGE_SIZE>(INITIAL_SIZE, pairs);
	tree_pair toAdd[] = {
		{ 0, 13 },
		{ 101, 7 },
		{ 32, 44 },
		{ 66, 99 } };
	for (auto& pair : toAdd)
	{
		BOOST_CHECK(tree.insert(pair.key, pair.value));
		pairs.push_back(pair);
		if (!check_if_all_key_exists(tree, pairs))
		{
			BOOST_CHECK_MESSAGE(false, "Test failed after insertion of (" << pair.key << ", " << pair.value << ").");
			return;
		}
		BOOST_CHECK_EQUAL(tree.get_height(), 0);
	}
}

BOOST_AUTO_TEST_CASE(cpu_insertion_to_root_with_node_spliting)
{
	const int PAGE_SIZE = 16;
	const int INITIAL_SIZE = 14;
	std::vector<tree_pair> pairs;
	auto tree = initialize_tree<PAGE_SIZE>(INITIAL_SIZE, pairs);
	tree_pair toAdd[] = {
		{ 32, 44 },
		{ 66, 99 },
		{ 52, 32 },
		{ 9, 13 },
		{ 13, 19 },
		{ 111, 3 },
		{ 123, 4 },
		{ 31, 2222 },
		{ 101, 7 }, };
	int inserted = 0;
	for (auto& pair : toAdd)
	{
		BOOST_CHECK(tree.insert(pair.key, pair.value));
		inserted += 1;
		pairs.push_back(pair);
		if (!check_if_all_key_exists(tree, pairs))
		{
			BOOST_CHECK_MESSAGE(false, "Test failed after insertion of (" << pair.key << ", " << pair.value << ").");
			return;
		}
		BOOST_CHECK_EQUAL(tree.get_height(), inserted >= 3 ? 1 : 0);
	}
}

BOOST_AUTO_TEST_CASE(cpu_insertion_to_leaf)
{
	const int PAGE_SIZE = 16;
	const int INITIAL_SIZE = 250;
	std::vector<tree_pair> pairs;
	auto tree = initialize_tree<PAGE_SIZE>(INITIAL_SIZE, pairs);
	const int toAdd = 1000;
	for (int i = toAdd; i > 0; --i)
	{
		int key = (950 - i) * 5 + 1;
		int value = i * 7;
		BOOST_CHECK(tree.insert(key, value));
		pairs.push_back(tree_pair{ key, value });
		if (!check_if_all_key_exists(tree, pairs))
		{
			BOOST_CHECK_MESSAGE(false, "Test failed after insertion of (" << key << ", " << value << ").");
			return;
		}
	}
}

#endif