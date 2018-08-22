#pragma once
#include "functions.h"
#include <algorithm>
#include "bplus_tree_gpu.cuh"
#include <iomanip>
#include <cctype>
#include <regex>
#include <set>
#include <sstream>
#include "dictionary_reader.h"
#include "book_reader.h"

template<typename HASH, int PAGE_SIZE, int Version>
void __test_gpu_tree_inner(const std::vector<HASH> &hashes, const std::vector<char> &dictSuffixes, const std::vector<int> &dictPos, const std::vector<char> &bookWords, const std::vector<int> &bookPos)
{
	bplus_tree_gpu<HASH, PAGE_SIZE> tree(hashes.data(), dictPos.data(), hashes.size(), dictSuffixes.data(), dictSuffixes.size());
	std::cout << "Tree created in " << std::setprecision(5) << tree.last_gpu_time() << " ms.\n";
	std::vector<bool> result = tree.template exist_word<Version>(bookWords.data(), bookWords.size(), bookPos.data(), bookPos.size());
	std::cout << "Search done in " << std::setprecision(5) << tree.last_gpu_time() << " ms.\n";
	auto found = std::count(result.begin(), result.end(), true);
	auto missed = result.size() - found;
	std::cout << "Found " << found << " out of " << result.size() << ". Missed " << missed << " words.\n"
		<< std::setw(6) << std::setprecision(5) << static_cast<float>(found) / result.size() << "% of words were not found.\n";
	std::stringstream stream;
	stream << "bplus_tree_gpu<" << typeid(HASH).name() << "; " << PAGE_SIZE << "> v" << Version;
	append_to_csv(stream.str().c_str(), NAN, tree.last_gpu_time(), hashes.size(), result.size(), static_cast<float>(found) / result.size());
}

template<typename HASH, int PAGE_SIZE, int Version>
void test_gpu_tree(dictionary_reader &dictReader, book_reader &bookReader)
{
	const std::vector<HASH> &hashes = dictReader.get_hashes<HASH>();
	auto tup = dictReader.get_suffixes<HASH>(1);
	const std::vector<char> &suffixes = std::get<0>(tup);
	const std::vector<int> &dictPos = std::get<1>(tup);
	constexpr int align = kernel_version_selector<HASH, PAGE_SIZE, Version>::wordsAlignment;
	auto tup2 = bookReader.get_words(align);
	const std::vector<char> &words = std::get<0>(tup2);
	const std::vector<int> &bookPos = std::get<1>(tup2);
	__test_gpu_tree_inner<HASH, PAGE_SIZE, Version>(hashes, suffixes, dictPos, words, bookPos);
}

template<typename HASH, int PAGE_SIZE, int Version>
void test_gpu_tree(const char* dictionaryFilename, const char* bookFilename)
{
	auto dictionaryArray = read_file_to_buffer(dictionaryFilename);
	std::replace(dictionaryArray.begin(), dictionaryArray.end(), '\n', '\0');
	std::vector<HASH> hashes;
	std::vector<char> suffixes;
	std::vector<int> positions;
	HASH lastHash = 0;
	int pos = 0;
	dictionaryArray.push_back('\0');
	for (auto dictIt = dictionaryArray.begin(); dictIt < dictionaryArray.end(); ++dictIt)
	{
		HASH newHash = get_hash<HASH>(&*dictIt, 0);
		if (lastHash != newHash)
		{
			positions.push_back(pos);
			hashes.push_back(newHash);
		}
		int len = std::strlen(&*dictIt);
		if (len > chars_in_type<HASH>)
		{
			suffixes.insert(suffixes.end(), dictIt + chars_in_type<HASH>, dictIt + len);
			pos += len - chars_in_type<HASH>;
		}
		dictIt += len;
		suffixes.push_back('\0');
		pos += 1;
		lastHash = newHash;
	}
	bplus_tree_gpu<HASH, PAGE_SIZE> tree(hashes.data(), positions.data(), hashes.size(), suffixes.data(), suffixes.size());
	std::cout << "Tree created in " << std::setprecision(5) << tree.last_gpu_time() << " ms.\n";

	auto bookArray = read_file_to_buffer(bookFilename);
	std::transform(bookArray.begin(), bookArray.end(), bookArray.begin(), [](char c) -> char {
		if (!std::isalpha(c))
			return ' ';
		return std::tolower(c);
	});
	std::replace_if(bookArray.begin(), bookArray.end(), [](char c) -> bool { return c == '\n' || c == '\r'; }, ' ');
	const std::regex wordRegex(R"(\b\w+\b)");
	std::vector<std::string> words;
	std::match_results<decltype(bookArray.begin())> match;
	auto beginSearch = bookArray.begin();
	while(std::regex_search(beginSearch, bookArray.end(), match, wordRegex))
	{
		auto str = match.str();
		if (str.length() > 0)
		{
			std::transform(str.begin(), str.end(), str.begin(), ::tolower);
			words.push_back(str);
		}
		beginSearch += str.length() + match.prefix().length();
	}
	std::string concated;
	std::vector<int> concatedPositions;
	pos = 0;
	for (auto& str : words)
	{
		concatedPositions.push_back(pos);
		concated.append(str);
		const int mod = kernel_version_selector<HASH, PAGE_SIZE, Version>::wordsAlignment;
		const int zeroes = mod == 1 ? 1 : mod - (str.size() % mod);
		concated.insert(concated.end(), zeroes, '\0');
		pos += str.size() + zeroes;
	}
	std::vector<bool> result = tree.template exist_word<Version>(concated.c_str(), concated.size(), concatedPositions.data(), concatedPositions.size());
	std::cout << "Search done in " << std::setprecision(5) << tree.last_gpu_time() << " ms.\n";
	auto found = std::count(result.begin(), result.end(), true);
	auto missed = result.size() - found;
	std::cout << "Found " << found << " out of " << result.size() << ". Missed " << missed << " words.\n"
		<< std::setw(6) << std::setprecision(5) << static_cast<float>(found) / result.size() << "% of words were not found.\n";
	std::stringstream stream;
	stream << "bplus_tree_gpu<" << typeid(HASH).name() << "; " << PAGE_SIZE << "> v" << Version;
	append_to_csv(stream.str().c_str(), NAN, tree.last_gpu_time(), hashes.size(), result.size(), static_cast<float>(found) / result.size());
	return;
}

