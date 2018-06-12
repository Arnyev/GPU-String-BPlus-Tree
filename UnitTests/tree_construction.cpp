#include "../bplus_tree_cpu.h"
#ifdef BOOST_ENABLE
#define BOOST_TEST_MODULE TreeConstruction
#include <boost/test/unit_test.hpp>

using HASH = int;

template<int PageSize>
void test_tree_creation(const int elements, const int excpectedHeight)
{
	std::vector<HASH> keys(elements);
	std::vector<int> values(elements);
	int k = 0;
	std::transform(keys.begin(), keys.end(), keys.begin(), [&k](HASH) -> HASH {return ++k; });
	k = elements + 1;
	std::transform(values.begin(), values.end(), values.begin(), [&k](HASH) -> HASH {return --k; });
	bplus_tree_cpu<HASH, PageSize> tree(keys.data(), values.data(), elements);
	for (k = 1; k <= elements; ++k)
	{
		int excpectedValue = elements - k + 1;
		int returnedValue = tree.get_value(k);
		BOOST_CHECK_EQUAL(excpectedValue, returnedValue);
		if (excpectedValue != returnedValue)
			break;
	}
	BOOST_CHECK_EQUAL(tree.get_height(), excpectedHeight);
}

template<int PageSize>
int compute_height(int elements)
{
	int height = 0;
	if (elements < PageSize)
		return height;
	elements = elements * 2 / PageSize;
	height += 1;
	while (elements > PageSize)
	{
		elements = elements / (PageSize / 2 + 1);
		height += 1;
	}
	return height;
}

BOOST_AUTO_TEST_CASE(tree_with_0_height)
{
	const int PAGE_SIZE = 16;
	int sizes[] = {
		PAGE_SIZE / 2,
		PAGE_SIZE / 4,
		1,
		PAGE_SIZE - 1};
	for (int size : sizes)
	{
		test_tree_creation<PAGE_SIZE>(size, 0);
	}
}

BOOST_AUTO_TEST_CASE(tree_with_1_height)
{
	const int PAGE_SIZE = 16;
	int sizes[] = {
		//Minimum for tree with height equal to 1
		PAGE_SIZE,
		PAGE_SIZE + PAGE_SIZE / 2,
		//Maximum for tree with height equal to 1
		(PAGE_SIZE / 2) * PAGE_SIZE + PAGE_SIZE - 1, };
	for (int size : sizes)
	{
		test_tree_creation<PAGE_SIZE>(size, 1);
	}
}


BOOST_AUTO_TEST_CASE(tree_with_2_height)
{
	const int PAGE_SIZE = 16;
	int sizes[] = {
		//Minimum for tree with height equal to 2
		(PAGE_SIZE / 2) * PAGE_SIZE + PAGE_SIZE,
		//Maximum for tree with height equal to 2
		((PAGE_SIZE / 2) * PAGE_SIZE + PAGE_SIZE - 1 + PAGE_SIZE + 1) * (PAGE_SIZE / 2) + PAGE_SIZE - 1};
	for (int size : sizes)
	{
		test_tree_creation<PAGE_SIZE>(size, 2);
	}
}

BOOST_AUTO_TEST_CASE(tree_with_height_greater_than_2)
{
	const int PAGE_SIZE = 16;
	int sizes[] = {
		2'000,
		5'000,
		7'777,
		12'000 };
	for (int size : sizes)
	{
		test_tree_creation<PAGE_SIZE>(size, compute_height<PAGE_SIZE>(size));
	}
}

#endif
