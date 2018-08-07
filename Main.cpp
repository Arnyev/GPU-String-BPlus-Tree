#include "bplus_tree_cpu.h"
#include "functions.h"
#include "sort_strings.cuh"
#include <helper_cuda.h>

using namespace std;

int main(const int argc, char **argv)
{
	using hash = ullong;
	std::vector<std::string> words;
	words.emplace_back("abc");
	words.emplace_back("axxxxxxxxxxxxtomek");
	words.emplace_back("axxxxxxxxxxxxjanek");
	words.emplace_back("domek");
	words.emplace_back("romek");
	words.emplace_back("xds");
	std::vector<hash> hashes;
	std::vector<int> indexes;
	std::vector<char> suffixes;
	std::string concat;
	std::vector<int> beginIndexes;
	const char nullbyte = static_cast<char>(0);
	int suffixIndex = 0;
	int nextIndex = 0;
	for (auto& word : words)
	{
		int length = 0;
		hash h = get_hash(reinterpret_cast<const uchar*>(word.c_str()), CHARSTOHASH, 0);
		if (word.length() > CHARSTOHASH)
		{
			for (auto it = word.begin() + CHARSTOHASH; it != word.end(); ++it)
			{
				suffixes.emplace_back(*it);
				++length;
			}
		}
		suffixes.emplace_back(nullbyte);
		++length;
		if (hashes.empty() || h != hashes.back())
		{
			hashes.emplace_back(h);
			indexes.push_back(suffixIndex);
		}
		suffixIndex += length;
		concat.append(word);
		concat.push_back(nullbyte);
		beginIndexes.emplace_back(nextIndex);
		nextIndex = static_cast<int>(word.size());
	}
	bplus_tree_cpu<hash, 4> tree(hashes.data(), indexes.data(), static_cast<int>(hashes.size()), suffixes.data(), static_cast<int>(suffixes.size()));
	tree.exist_word("axxxxxxxxxxxxjanex");
	bplus_tree_gpu<hash, 4> gtree(hashes.data(), indexes.data(), hashes.size(), suffixes.data(), suffixes.size());
	//char *toFind = "domek\0romek";
	//int tab[2] = { 0, 6 };
	//gtree.exist_word(toFind, 12, tab, 2);
	//Sample creation of B+ tree
	//int i = 0;
	//int size = 64;
	//vector<int> keys(size);
	//vector<int> values(size);
	//generate(keys.begin(), keys.end(), [&i]() -> int { return ++i; });
	//i = 1;
	//generate(values.begin(), values.end(), [&i]() -> int { return ++i; });
	//bplus_tree_gpu<int, 4> gt(keys.data(), values.data(), size);
	//bplus_tree_cpu<int, 4> ct(gt);
	//bplus_tree_cpu<int, 4> cte(keys.data(), values.data(), size);
	//int toFind[] = { 1, 2, 0, -1, 64, 65, 3 };
	//auto z1 = ct.get_value(toFind, sizeof(toFind) / sizeof(int));
	//auto z2 = cte.get_value(toFind, sizeof(toFind) / sizeof(int));
	//auto z3 = gt.get_value(toFind, sizeof(toFind) / sizeof(int));
	//ct and cte should be equal

	findCudaDevice(argc, const_cast<const char **>(argv));

	int* test;//initialization to improve time testing accuracy
	if (cudaMalloc(&test, 4 * 4))
		return 0;

	test_array_searching_book("dictionary.txt", "oliverTwist.txt");
	test_array_searching_book("dictionary.txt", "book.txt");

	cout << "Randoms" << endl;
	test_random_strings();

	cout << "Moby Dick" << endl;
	test_book("book.txt");
}
