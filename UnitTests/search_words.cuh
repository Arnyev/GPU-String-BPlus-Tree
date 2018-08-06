#pragma once
#include <set>
#include <string>
#include "../bplus_tree_gpu.cuh"
#include "../sort_helpers.cuh"
#include "../parameters.h"
#include <tuple>
#include <numeric>

using HASH = unsigned long long;
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
	int pos = 0;
	for (auto& str : words)
	{
		positions.push_back(pos);
		if (str.size() > chars_in_type<HASH>)
		{
			suffixes.insert(suffixes.end(), str.begin() + chars_in_type<HASH>, str.end());
			pos += str.size() - chars_in_type<HASH>;
		}
		suffixes.push_back('\0');
		pos += 1;
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

auto sort_by_hashes(std::vector<std::string> &words, std::vector<HASH> &hashes)
{
	std::vector<int> indexes(words.size());
	std::iota(indexes.begin(), indexes.end(), 0);
	//std::sort(indexes.begin(), indexes.end(), [](int a, int b) -> int {
	//	return hashes[b] - hashes[a];
	//})
}

bool find_words_tmp()
{
	std::set<std::string> set_words;
	set_words.emplace("bonek");
	set_words.emplace("domek");
	set_words.emplace("romek");
	set_words.emplace("abomekkkkkkkkkx");
	std::vector<std::string> words(set_words.begin(), set_words.end());
	std::vector<HASH> hashes = get_hashes(words);
	std::vector<char> suffixes;
	std::vector<int> positions;
	std::tie(suffixes, positions) = get_suffixes(words);
	bplus_tree_gpu<HASH, PAGE_SIZE> tree(hashes.data(), positions.data(), words.size(), suffixes.data(), suffixes.size());
	std::string concated;
	std::vector<int> positionsInConcat;
	std::vector<std::string> to_search(words);
	to_search.insert(to_search.begin() + 2, "tomek");
	to_search.insert(to_search.begin() + 4, "abomekkkkkkkkkx");
	to_search.insert(to_search.begin() + 5, "abomekkkkkkkkkz");
	std::tie(concated, positionsInConcat) = concat_words(to_search);
	auto result = tree.exist_word(concated.c_str(), concated.size(), positionsInConcat.data(), positionsInConcat.size());
	return result[0] &&
		result[1] && 
		!result[2] &&
		result[3] &&
		result[4] &&
		!result[5] &&
		result[6];
}
