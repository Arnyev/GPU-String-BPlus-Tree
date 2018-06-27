#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <helper_math.h>

typedef unsigned char uchar;
typedef unsigned long long ullong;
#define BREAKCHAR ' '

using namespace std;

inline bool test_string_sorting(int * h_word_positions, int * d_output, const int word_count, 
	unsigned char * h_word_array, uchar * word_array_out, const size_t word_array_size)
{
	checkCudaErrors(cudaMemcpy(h_word_positions, d_output, word_count * sizeof(int), cudaMemcpyDeviceToHost));
	int word_array_index = 0;

	const string words(h_word_array, h_word_array + word_array_size);
	istringstream iss_words(words);
	vector<string> words_split(istream_iterator<string>{iss_words}, istream_iterator<string>());

	for (auto& i : words_split)
		std::transform(i.begin(), i.end(), i.begin(), ::tolower);

	std::sort(words_split.begin(), words_split.end());

	const auto it = unique(words_split.begin(), words_split.end());
	words_split.resize(std::distance(words_split.begin(), it));

	std::sort(words_split.begin(), words_split.end());

	vector<int> positions(h_word_positions, h_word_positions + word_count);

	for (int i = 0; i < word_count; i++)
	{
		const int position = positions[i];
		if (position == -1)
			continue;
		int index_in_word = 0;
		while (true)
		{
			const uchar c = h_word_array[position + index_in_word];
			if (c != BREAKCHAR)
			{
				word_array_out[word_array_index++] = c;
				index_in_word++;
			}
			else
				break;
		}
		word_array_out[word_array_index++] = BREAKCHAR;
	}

	const string words_sorted_gpu(word_array_out, word_array_out + word_array_index);
	istringstream iss_words_gpu(words_sorted_gpu);

	vector<string> words_sorted_split_gpu(istream_iterator<string>{iss_words_gpu}, istream_iterator<string>());

	int fails = 0;
	for (uint i = 0; i < words_sorted_split_gpu.size(); i++)
	{
		std::transform(words_sorted_split_gpu[i].begin(), words_sorted_split_gpu[i].end(), words_sorted_split_gpu[i].begin(), ::tolower);
		string s1 = words_sorted_split_gpu[i];
		string s2 = words_split[i];
		string prev1, prev2, next1, next2;
		if(i>0)
		{
			prev1 = words_sorted_split_gpu[i - 1];
			prev2 = words_split[i - 1];
		}

		if(i<words_sorted_split_gpu.size()-1)
		{
			next1 = words_sorted_split_gpu[i + 1];
			next2 = words_split[i + 1];
		}

		if (words_sorted_split_gpu[i] != words_split[i])
			fails++;
	}

	return fails == 0;
}
