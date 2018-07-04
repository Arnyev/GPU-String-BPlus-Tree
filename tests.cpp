#include <vector>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <helper_math.h>
#include <numeric>
#include "parameters.h"
#include "helpers.h"
#include "functions.h"
#include <iostream>

using namespace std;

vector<string> get_sorted_cpu_words(unsigned char* h_word_array, const size_t word_array_size)
{
	vector<uchar> wordsVec(h_word_array, h_word_array + word_array_size);
	for (auto& i : wordsVec)
		if (i == BREAKCHAR)
			i = SPLITTER;

	const string words(wordsVec.begin(), wordsVec.end());

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
		word_array_out[word_array_index++] = SPLITTER;
	}

	const string words_sorted_gpu(word_array_out, word_array_out + word_array_index);
	istringstream iss_words_gpu(words_sorted_gpu);

	return  vector<string>(istream_iterator<string>{iss_words_gpu}, istream_iterator<string>());
}


void generate_random_strings(int* h_positions, unsigned char* h_chars, int& cur_char_index, int& cur_word_index, const int chars_count)
{
	cur_char_index = 0;
	cur_word_index = 0;
	for (; cur_word_index < RANDSTRCOUNT; cur_word_index++)
	{
		if (cur_char_index > chars_count - RANDSTRMAXLEN - 1)
			break;

		h_positions[cur_word_index] = cur_char_index;
		const int strlen = rand() % RANDSTRMAXLEN + 1;
		for (int j = 0; j < strlen; j++)
			h_chars[cur_char_index++] = RANDCHARSET[rand() % RANDCHARSCOUNT];

		h_chars[cur_char_index++] = BREAKCHAR;
	}
}

bool test_random_strings()
{
	const auto h_positions = static_cast<int*>(malloc(RANDSTRCOUNT * sizeof(int)));
	const int chars_count = RANDSTRCOUNT * RANDSTRMAXLEN / 2;
	const auto h_chars = static_cast<unsigned char*>(malloc(chars_count * sizeof(char)));
	int cur_char_index;
	int cur_word_index;
	generate_random_strings(h_positions, h_chars, cur_char_index, cur_word_index, chars_count);

	unsigned char* d_word_array;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_word_array), cur_char_index + CHARSTOHASH));
	checkCudaErrors(cudaMemcpy(d_word_array, h_chars, cur_char_index, cudaMemcpyHostToDevice));
	int* d_word_positions;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_word_positions), sizeof(int)*cur_word_index));
	checkCudaErrors(cudaMemcpy(d_word_positions, h_positions, sizeof(int)*cur_word_index, cudaMemcpyHostToDevice));

	const auto d_sorted_positions = get_sorted_positions(d_word_positions, cur_word_index, d_word_array);
	const bool sorting_result = test_string_sorting(d_sorted_positions, cur_word_index, h_chars, cur_char_index);

	const auto output = create_output(d_word_array, d_sorted_positions, cur_word_index);
	const bool output_result = test_output(h_chars, cur_char_index, output);

	free(h_positions);
	free(h_chars);

	checkCudaErrors(cudaFree(d_word_array));
	checkCudaErrors(cudaFree(d_word_positions));

	return output_result && sorting_result;
}

bool test_string_sorting(int * d_sorted_positions, const int word_count, unsigned char * h_word_array, const size_t word_array_size)
{
	auto words_cpu = get_sorted_cpu_words(h_word_array, word_array_size);
	auto words_gpu = get_sorted_gpu_words(d_sorted_positions, word_count, h_word_array, word_array_size);

	for (auto& i : words_gpu)
		std::transform(i.begin(), i.end(), i.begin(), ::tolower);

	for (uint i = 0; i < words_gpu.size(); i++)
	{
		if (words_gpu[i] != words_cpu[i])
		{
			cout << "Bad sort" << endl;
			return false;
		}
	}
	return true;
}

bool test_output(unsigned char* h_word_array, const int chars_input_count, const sorting_output output)
{
	auto words_split = get_sorted_cpu_words(h_word_array, chars_input_count);

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
	if (hashes.size() != static_cast<ullong>(output.hashes_count))
	{
		cout << "Bad hashes count" << endl;
		fail = true;
	}

	for (uint i = 0; i < d_hash.size(); i++)
		if (d_hash[i] != hashes[i])
		{
			cout << "Bad hash" << endl;
			fail = true;
			break;
		}

	vector<int> lens(words_split.size());
	std::transform(words_split.begin(), words_split.end(), lens.begin(), postfix_len_from_str);

	const auto lenchars = std::accumulate(lens.begin(), lens.end(), 0);
	if (output.suffixes_size != lenchars)
	{
		cout << "Bad suffix size" << endl;
		fail = true;
	}

	auto za = create_vector(output.suffixes, output.suffixes_size);

	std::transform(za.begin(), za.end(), za.begin(), ::tolower);

	for (int i = 0; i < lenchars; i++)
		if (chars[i] != za[i])
		{
			cout << "Bad char" << endl;
			fail = true;
			break;
		}

	auto vaa = create_vector(output.positions, output.hashes_count);
	for (int i = 0; i < output.hashes_count; i++)
		if (vaa[i] != positions[i])
		{
			cout << "Bad position" << endl;
			fail = true;
			break;
		}

	return !fail;
}
