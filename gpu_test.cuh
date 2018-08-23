#pragma once
#include <algorithm>
#include <iomanip>
#include <sstream>

#include "bplus_tree_gpu.cuh"
#include "csv_logger.h"
#include "dictionary_reader.h"
#include "book_reader.h"

template<typename HASH, int PAGE_SIZE, int Version>
void test_gpu_tree(dictionary_reader &dictReader, book_reader &bookReader, csv_logger &logger)
{
	float preprocTime, execTime, postTime;
	//Preprocesing on CPU
	auto start = std::chrono::steady_clock::now();
	const std::vector<HASH> &hashes = dictReader.get_hashes<HASH>(false);
	auto tup = dictReader.get_suffixes<HASH>(1, false);
	const std::vector<char> &dictSuffixes = std::get<0>(tup);
	const std::vector<int> &dictPos = std::get<1>(tup);
	constexpr int align = kernel_version_selector<HASH, PAGE_SIZE, Version>::wordsAlignment;
	auto tup2 = bookReader.get_words(align, false);
	const std::vector<char> &bookWords = std::get<0>(tup2);
	const std::vector<int> &bookPos = std::get<1>(tup2);
	auto duration = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
	//Tree creation
	bplus_tree_gpu<HASH, PAGE_SIZE> tree(hashes.data(), dictPos.data(), hashes.size(), dictSuffixes.data(), dictSuffixes.size());
	//Search algorithm
	std::vector<bool> result = tree.template exist_word<Version>(bookWords.data(), bookWords.size(), bookPos.data(), bookPos.size(), preprocTime, execTime, postTime);
	preprocTime += preprocTime;
	//Counting results
	auto found = std::count(result.begin(), result.end(), true);
	auto missed = result.size() - found;
	//Logging result
	std::stringstream stream;
	stream << "bplus_tree_gpu<" << typeid(HASH).name() << "; " << PAGE_SIZE << "> v" << Version;
	logger.append(stream.str().c_str(), NAN, preprocTime, execTime, postTime, dictReader.words_count(), bookReader.words_count(), static_cast<float>(found) / bookReader.words_count());
}
