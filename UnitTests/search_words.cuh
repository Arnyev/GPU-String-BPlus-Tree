#pragma once
#include <set>
#include <string>
#include "../bplus_tree_gpu.cuh"
#include "../sort_helpers.cuh"
#include "../parameters.h"
#include <tuple>
#include <numeric>

using HASH = unsigned long long;
using words_list = std::vector<std::string>;
using missing_tuple = std::tuple<int, std::string>;
using missing_words_list = std::vector<missing_tuple>;
const int PAGE_SIZE = 4;

std::vector<HASH> get_hashes(std::vector<std::string>& words)
{
	std::vector<HASH> result(words.size());
	std::transform(words.begin(), words.end(), result.begin(), [](std::string& word) -> HASH { return get_hash<HASH>(word.c_str(), 0); });
	return result;
}

auto get_suffixes(std::vector<std::string>& words)
{
	std::vector<char> suffixes;
	std::vector<int> positions;
	HASH lastHash = 0;
	int pos = 0;
	for (auto& str : words)
	{
		HASH newHash = get_hash<HASH>(str.c_str(), 0);
		if (lastHash != newHash)
		{
			positions.push_back(pos);
		}
		if (str.size() > chars_in_type<HASH>)
		{
			suffixes.insert(suffixes.end(), str.begin() + chars_in_type<HASH>, str.end());
			pos += str.size() - chars_in_type<HASH>;
		}
		suffixes.push_back('\0');
		pos += 1;
		lastHash = newHash;
	}
	return std::make_tuple(suffixes, positions);
}

auto concat_words(std::vector<std::string> &vect)
{
	std::string result;
	std::vector<int> positions;
	int pos = 0;
	for (auto& str : vect)
	{
		positions.push_back(pos);
		result.append(str);
		result.push_back('\0');
		pos += str.size() + 1;
	}
	return std::make_tuple(result, positions);
}

void remove_duplicates(std::vector<HASH> &hashes)
{
	hashes.resize(std::distance(hashes.begin(), std::unique(hashes.begin(), hashes.end())));
}

auto sort_by_hashes(std::vector<std::string> &words, std::vector<HASH> &hashes)
{
	std::vector<int> indexes(words.size());
	std::iota(indexes.begin(), indexes.end(), 0);
	std::sort(indexes.begin(), indexes.end(), [&hashes](int a, int b) -> int {
		return hashes[a] < hashes[b];
	});
	std::vector<HASH> sortedHashes(hashes.size());
	std::vector<std::string> sortedWords(words.size());
	int k = 0;
	for (auto i : indexes)
	{
		sortedHashes[k] = hashes[i];
		sortedWords[k] = words[i];
		++k;
	}
	words.swap(sortedWords);
	hashes.swap(sortedHashes);
	return;
}

bool do_test(words_list &words, missing_words_list &missingWords)
{
	std::vector<HASH> hashes = get_hashes(words);
	sort_by_hashes(words, hashes);
	remove_duplicates(hashes);
	std::vector<char> suffixes;
	std::vector<int> positions;
	std::tie(suffixes, positions) = get_suffixes(words);
	bplus_tree_gpu<HASH, PAGE_SIZE> tree(hashes.data(), positions.data(), hashes.size(), suffixes.data(), suffixes.size());
	std::vector<std::string> toSearch(words);
	std::vector<int> _failIndexes(missingWords.size());
	std::transform(missingWords.begin(), missingWords.end(), _failIndexes.begin(), [](auto &t) -> int { return std::get<0>(t); });
	std::set<int> failIndexes(_failIndexes.begin(), _failIndexes.end());
	int idx;
	std::string str;
	for (auto && tup : missingWords)
	{
		std::tie(idx, str) = tup;
		toSearch.insert(toSearch.begin() + idx, str);
	}
	std::string concated;
	std::vector<int> positionsInConcat;
	std::tie(concated, positionsInConcat) = concat_words(toSearch);
	auto result = tree.exist_word(concated.c_str(), concated.size(), positionsInConcat.data(), positionsInConcat.size());
	for (int i = 0; i < result.size(); ++i)
	{
		if ((failIndexes.find(i) != failIndexes.end()) == (result[i]))
			return false;
	}
	return true;
}

bool less_than_14_chars_1()
{
	words_list words{
		"maupassant",
		"lateral",
		"tooth",
		"mungo",
		"reform",
		"ryal",
		"nonsailor",
		"elias",
		"kennelled",
		"magdalen",
	};
	missing_words_list toMiss;
	return do_test(words, toMiss);
}

bool less_than_14_chars_2()
{
	words_list words{
		"unfundamental",
		"msting",
		"unexceptional",
		"drosometer",
		"circler",
		"dillie",
		"beguiler",
		"nonchurchgoer",
		"ambroise",
		"superordain",
		"marlin",
		"anaesthetized",
		"scofflaw",
		"dichromat",
		"khafre",
		"ephemerally",
		"unfunded",
		"peptidase",
		"compaction",
		"swinburne"
	};
	missing_words_list toMiss{
		missing_tuple(2, "vasopressin"),
		missing_tuple(4, "redelete"),
		missing_tuple(5, "mazzard"),
		missing_tuple(12, "swat"),
		missing_tuple(13, "neopilina")
	};
	return do_test(words, toMiss);
}

bool repeating_prefixes_1()
{
	words_list words{
		"aaaaaaaaaaaaxaaa", "aaaaaaaaaaaaxbbb", "aaaaaaaaaaaaxccc", "aaaaaaaaaaaaxddd", "aaaaaaaaaaaaxeee",
		"bbbbbbbbbbbbxaaa", "bbbbbbbbbbbbxbbb", "bbbbbbbbbbbbxccc", "bbbbbbbbbbbbxddd", "bbbbbbbbbbbbxeee",
		"ccccccccccccxaaa", "ccccccccccccxbbb", "ccccccccccccxccc", "ccccccccccccxddd", "ccccccccccccxeee",
		"ddddddddddddxaaa", "ddddddddddddxbbb", "ddddddddddddxccc", "ddddddddddddxddd", "ddddddddddddxeee",
		"eeeeeeeeeeeexaaa", "eeeeeeeeeeeexbbb", "eeeeeeeeeeeexccc", "eeeeeeeeeeeexddd", "eeeeeeeeeeeexeee",
		"ffffffffffffxaaa", "ffffffffffffxbbb", "ffffffffffffxccc", "ffffffffffffxddd", "ffffffffffffxeee",
		"ggggggggggggxaaa", "ggggggggggggxbbb", "ggggggggggggxccc", "ggggggggggggxddd", "ggggggggggggxeee",
		"hhhhhhhhhhhhxaaa", "hhhhhhhhhhhhxbbb", "hhhhhhhhhhhhxccc", "hhhhhhhhhhhhxddd", "hhhhhhhhhhhhxeee",
	};
	missing_words_list toMiss;
	return do_test(words, toMiss);
}

bool repeating_prefixes_2()
{
	words_list words{
		"aaaaaaaaaaaaxaaa", "aaaaaaaaaaaaxbbb", "aaaaaaaaaaaaxccc", "aaaaaaaaaaaaxddd", "aaaaaaaaaaaaxeee",
		"bbbbbbbbbbbbxaaa", "bbbbbbbbbbbbxbbb", "bbbbbbbbbbbbxccc", "bbbbbbbbbbbbxddd", "bbbbbbbbbbbbxeee",
		"ccccccccccccxaaa", "ccccccccccccxbbb", "ccccccccccccxccc", "ccccccccccccxddd", "ccccccccccccxeee",
		"ddddddddddddxaaa", "ddddddddddddxbbb", "ddddddddddddxccc", "ddddddddddddxddd", "ddddddddddddxeee",
		"eeeeeeeeeeeexaaa", "eeeeeeeeeeeexbbb", "eeeeeeeeeeeexccc", "eeeeeeeeeeeexddd", "eeeeeeeeeeeexeee",
		"ffffffffffffxaaa", "ffffffffffffxbbb", "ffffffffffffxccc", "ffffffffffffxddd", "ffffffffffffxeee",
		"ggggggggggggxaaa", "ggggggggggggxbbb", "ggggggggggggxccc", "ggggggggggggxddd", "ggggggggggggxeee",
		"hhhhhhhhhhhhxaaa", "hhhhhhhhhhhhxbbb", "hhhhhhhhhhhhxccc", "hhhhhhhhhhhhxddd", "hhhhhhhhhhhhxeee",
	};
	missing_words_list toMiss{
		missing_tuple(4, "aaaaaaaaaaaaxaxa"),
		missing_tuple(8, "bbbbbbbbbbbbxfff"),
		missing_tuple(13, "ccccccccccccx"),
		missing_tuple(14, "ddddddddddddxa"),
		missing_tuple(19, "ffffffffffffxcccc"),
		missing_tuple(23, "ggggggggggggxeea"),
	};
	return do_test(words, toMiss);
}
