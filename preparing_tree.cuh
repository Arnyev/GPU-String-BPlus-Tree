#pragma once
#include <thrust/transform_scan.h>
#include <thrust/device_vector.h>
#include "sort_helpers.cuh"
#include <algorithm>
#include "functions.h"
#include <thrust/unique.h>

template <class HASH>
void create_output(const thrust::device_vector<char>& words, thrust::device_vector<int>& sorted_positions, thrust::device_vector<HASH>& hashes,
                   thrust::device_vector<int>& positions, thrust::device_vector<char>& suffixes)
{
	auto word_count = sorted_positions.size();
	sorted_positions.resize(word_count + 1);
	positions.resize(word_count + 1);

	const compute_postfix_length_functor postfix_functor(words.data().get());
	transform_exclusive_scan(sorted_positions.begin(), sorted_positions.end(), positions.begin(), postfix_functor, 0, thrust::plus<int>());

	const int output_size = positions.back();

	suffixes.resize(output_size);

	copy_suffixes(words, sorted_positions, word_count, positions, suffixes);
	hashes.resize(word_count);
	auto functor = hash_functor<HASH>(words.data().get());

	thrust::transform(sorted_positions.begin(), sorted_positions.begin() + word_count, hashes.begin(), functor);

	const auto hashes_end = thrust::unique_by_key(hashes.begin(), hashes.end(), positions.begin());
	const auto hashes_count = hashes_end.first - hashes.begin();
	hashes.resize(hashes_count);
	positions.resize(hashes_count);
}

template <class HASH>
void create_output_cpu(const thrust::host_vector<char>& words, const thrust::host_vector<int>& words_positions,
                       std::vector<HASH>& hashes, std::vector<int>& positions, std::vector<char>& suffixes)
{
	std::vector<std::string> words_split;
	get_sorted_unique_cpu_strings(words, words_positions, words_split);

	hashes.resize(words_split.size());

	std::transform(words_split.begin(), words_split.end(), hashes.begin(), get_hash_str<HASH>);

	for (uint i = 0; i < words_split.size(); i++)
	{
		auto sts = words_split[i];
		if (i == 0 || hashes[i] != hashes[i - 1])
			positions.push_back(static_cast<int>(suffixes.size()));

		if (sts.size() >= CHARSTOHASH)
		{
			for (uint j = CHARSTOHASH; j < sts.size(); j++)
				suffixes.push_back(sts[j]);
			suffixes.push_back(BREAKCHAR);
		}
	}

	const auto hashes_end = std::unique(hashes.begin(), hashes.end());
	hashes.resize(std::distance(hashes.begin(), hashes_end));
}
