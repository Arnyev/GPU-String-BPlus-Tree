#pragma once
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "sort_helpers.cuh"

class dictionary_reader
{
	std::vector<std::string> words;
	std::unordered_map<std::string, std::tuple<std::vector<char>, std::vector<int>>> suffixes;
	std::unordered_map<std::string, void*> hashes;
public:
	dictionary_reader(const char* fileName);
	dictionary_reader(const std::string &fileName);
	dictionary_reader(const std::string &&fileName);

	template<typename HASH>
	const std::vector<HASH>& get_hashes(bool useCache = true);
	
	template<typename HASH>
	std::tuple<const std::vector<char>&, const std::vector<int>&> get_suffixes(int align, bool useCache = true);
};

template <typename HASH>
const std::vector<HASH>& dictionary_reader::get_hashes(bool useCache)
{
	auto it = hashes.find(typeid(HASH).name());
	if (useCache && it != hashes.end())
	{
		return *reinterpret_cast<std::vector<HASH>*>(it->second);
	}
	auto b = hashes.emplace(std::string(typeid(HASH).name()), new std::vector<HASH>()).first->second;
	std::vector<HASH>& result = *reinterpret_cast<std::vector<HASH>*>(hashes.emplace(std::string(typeid(HASH).name()), new std::vector<HASH>()).first->second);
	result.resize(words.size());
	std::transform(words.begin(), words.end(), result.begin(), [](std::string& str) -> HASH { return get_hash_v2<HASH>(str.c_str(), 0); });
	result.resize(std::distance(result.begin(), std::unique(result.begin(), result.end())));
	return result;
}

template <typename HASH>
std::tuple<const std::vector<char>&, const std::vector<int>&> dictionary_reader::get_suffixes(int align, bool useCache)
{
	assert(align >= 1);
	std::string name = std::to_string(align) + "&" + std::string(typeid(HASH).name());
	auto it = suffixes.find(name);
	if (useCache && it != suffixes.end())
	{
		return it->second;
	}
	auto inserted = suffixes.emplace(name, std::make_tuple(std::vector<char>(), std::vector<int>()));
	std::vector<char> &result = std::get<0>(inserted.first->second);
	std::vector<int> &positions = std::get<1>(inserted.first->second);
	HASH lastHash = 0;
	for (auto & word : words)
	{
		HASH newHash = get_hash_v2<HASH>(word.c_str(), 0);
		const int toSkip = std::min(static_cast<int>(word.size()), chars_in_type<HASH>);
		const int toInsert = word.size() - toSkip;
		if (lastHash != newHash)
		{
			positions.emplace_back(result.size());
		}
		result.insert(result.end(), word.begin() + toSkip, word.end());
		const int zeros = align - (toInsert % align);
		result.insert(result.end(), zeros, '\0');
		lastHash = newHash;
	}
	return {result, positions};
}
