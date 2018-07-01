#include <vector>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <helper_math.h>
#include <numeric>
#include "parameters.h"
#include "helpers.h"
#include "functions.h"

using namespace std;

vector<string> get_sorted_cpu_words(unsigned char* h_word_array, const size_t word_array_size)
{
	const string words(h_word_array, h_word_array + word_array_size);
	istringstream iss_words(words);
	auto words_split = vector<string>(istream_iterator<string>{iss_words}, istream_iterator<string>());

	for (auto& i : words_split)
		std::transform(i.begin(), i.end(), i.begin(), ::tolower);

	std::sort(words_split.begin(), words_split.end());

	const auto it = unique(words_split.begin(), words_split.end());
	words_split.resize(std::distance(words_split.begin(), it));

	return words_split;
}

vector<string> get_sorted_gpu_words(int* d_positions, const int word_count, const unsigned char* h_word_array,
	const size_t word_array_size)
{
	auto positions = create_vector(d_positions, word_count);
	const auto word_array_out = reinterpret_cast<uchar*>(malloc(word_array_size * sizeof(uchar)));
	int word_array_index = 0;

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

	return  vector<string>(istream_iterator<string>{iss_words_gpu}, istream_iterator<string>());
}

bool test_string_sorting(int * d_positions, const int word_count, unsigned char * h_word_array, const size_t word_array_size)
{
	auto words_cpu = get_sorted_cpu_words(h_word_array, word_array_size);
	auto words_gpu = get_sorted_gpu_words(d_positions, word_count, h_word_array, word_array_size);

	int fails = 0;
	for (auto& i : words_gpu)
		std::transform(i.begin(), i.end(), i.begin(), ::tolower);

	for (uint i = 0; i < words_gpu.size(); i++)
		if (words_gpu[i] != words_cpu[i])
			fails++;

	return fails == 0;
}

bool test_output(unsigned char* d_word_array, const int chars_input_count, const sorting_output output)
{
	vector<unsigned char> initial_words = create_vector(d_word_array, chars_input_count);
	const string words(initial_words.begin(), initial_words.end());
	istringstream iss_words(words);
	vector<string> words_split(istream_iterator<string>{iss_words}, istream_iterator<string>());

	for (auto& i : words_split)
		std::transform(i.begin(), i.end(), i.begin(), ::tolower);

	std::sort(words_split.begin(), words_split.end());

	const auto it = std::unique(words_split.begin(), words_split.end());
	words_split.resize(std::distance(words_split.begin(), it));

	std::sort(words_split.begin(), words_split.end());

	vector<ullong> hashes(words_split.size());
	std::transform(words_split.begin(), words_split.end(), hashes.begin(), cpu_hash);

	vector<char> chars(output.suffixes_size);
	int cur_char_position = 0;
	vector<int> positions{};
	for (uint i = 0; i < words_split.size(); i++)
	{
		auto sts = words_split[i];
		if (i == 0 || hashes[i] != hashes[i - 1])
			positions.push_back(cur_char_position);

		if (sts.size() >= CHARSTOHASH)
		{
			for (uint j = CHARSTOHASH; j < sts.size(); j++)
				chars[cur_char_position++] = sts[j];
			chars[cur_char_position++] = 0;
		}
	}

	bool fail = false;
	auto d_hash = create_vector(output.hashes, output.hashes_count);
	const auto ita = std::unique(hashes.begin(), hashes.end());
	hashes.resize(std::distance(hashes.begin(), ita));

	for (uint i = 0; i < d_hash.size(); i++)
		if (d_hash[i] != hashes[i])
			fail = true;

	vector<int> lens(words_split.size());
	std::transform(words_split.begin(), words_split.end(), lens.begin(), postfix_len_from_str);

	const auto lenchars = std::accumulate(lens.begin(), lens.end(), 0);
	if (output.suffixes_size != lenchars)
		fail = true;

	auto za = create_vector(output.suffixes, output.suffixes_size);

	for (char& c : chars)
		if (c == 0)
			c = BREAKCHAR;

	std::transform(za.begin(), za.end(), za.begin(), ::tolower);

	for (int i = 0; i < lenchars; i++)
		if (chars[i] != za[i])
			fail = true;

	auto vaa = create_vector(output.positions, output.hashes_count);
	for (int i = 0; i < output.hashes_count; i++)
		if (vaa[i] != positions[i])
			fail = true;

	return !fail;
}
