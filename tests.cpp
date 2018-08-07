#include <algorithm>
#include <string>
#include <unordered_set>
#include "functions.h"

using namespace std;

vector<string> get_sorted_cpu_words(const thrust::host_vector<uchar>& words_chars)
{
	vector<char> buf;
	vector<string> words;
	for (size_t i = 0; i < words_chars.size(); i++)
	{
		const auto c = words_chars[i];
		if (c == BREAKCHAR)
		{
			if (!buf.empty())
			{
				words.emplace_back(buf.begin(), buf.end());
				buf.clear();
			}
		}
		else
			buf.push_back(c);
	}
	sort(words.begin(), words.end());

	return words;
}

vector<string> get_sorted_unique_cpu_words(const thrust::host_vector<uchar> & words_chars)
{
	auto words = get_sorted_cpu_words(words_chars);

	const auto it = unique(words.begin(), words.end());
	words.resize(std::distance(words.begin(), it));

	return words;
}

vector<string> get_sorted_gpu_words(const thrust::device_vector<int>& sorted_positions, const thrust::host_vector<uchar> & words)
{
	thrust::host_vector<int> positions(sorted_positions);
	vector<string> result;
	vector<char> word;

	for (size_t i = 0; i < sorted_positions.size(); i++)
	{
		const int position = positions[i];
		if (position == -1)
			continue;

		int index_in_word = 0;
		while (true)
		{
			const uchar c = words[position + index_in_word];
			if (c != BREAKCHAR)
			{
				word.push_back(c);
				index_in_word++;
			}
			else
				break;
		}

		result.emplace_back(word.begin(), word.end());
		word.clear();
	}

	return result;
}

bool test_string_sorting(const thrust::device_vector<int>& sorted_positions, const thrust::host_vector<uchar> & words)
{
	auto words_gpu = get_sorted_gpu_words(sorted_positions, words);
	auto words_cpu = get_sorted_unique_cpu_words(words);

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

ullong cpu_hash(std::string s)
{
	int i = 0;
	ullong hash = 0;
	for (; i < CHARSTOHASH; i++)
	{
		const unsigned char c = s[i];
		if (c == '\0')
			break;

		hash *= ALPHABETSIZE;
		hash += c & CHARMASK;
	}

	const ullong mask = s[i] == '\0' ? 0 : 1;

	for (; i < CHARSTOHASH; i++)
		hash *= ALPHABETSIZE;

	hash <<= 1;
	hash |= mask;
	return hash;
}

sorting_output_cpu get_cpu_output(const thrust::host_vector<uchar> & words)
{
	sorting_output_cpu output;

	auto words_split = get_sorted_unique_cpu_words(words);

	output.hashes.resize(words_split.size());
	std::transform(words_split.begin(), words_split.end(), output.hashes.begin(), cpu_hash);

	for (uint i = 0; i < words_split.size(); i++)
	{
		auto sts = words_split[i];
		if (i == 0 || output.hashes[i] != output.hashes[i - 1])
			output.positions.push_back(static_cast<int>(output.suffixes.size()));

		if (sts.size() >= CHARSTOHASH)
		{
			for (uint j = CHARSTOHASH; j < sts.size(); j++)
				output.suffixes.push_back(sts[j]);
			output.suffixes.push_back(BREAKCHAR);
		}
	}

	const auto hashes_end = std::unique(output.hashes.begin(), output.hashes.end());
	output.hashes.resize(std::distance(output.hashes.begin(), hashes_end));

	return output;
}

bool test_output(const thrust::host_vector<uchar> & words, const sorting_output_gpu& gpu_output)
{
	const auto cpu_output = get_cpu_output(words);

	if (cpu_output.hashes.size() != gpu_output.hashes.size())
	{
		cout << "Bad hashes count" << endl;
		return false;
	}

	thrust::host_vector<ullong> gpu_hashes(gpu_output.hashes);
	for (size_t i = 0; i < gpu_hashes.size(); i++)
		if (gpu_hashes[i] != cpu_output.hashes[i])
		{
			cout << "Bad hash" << endl;
			return false;
		}

	if (gpu_output.suffixes.size() != cpu_output.suffixes.size())
	{
		cout << "Bad suffix size" << endl;
		return false;
	}

	thrust::host_vector<uchar> gpu_suffixes(gpu_output.suffixes);
	for (size_t i = 0; i < gpu_suffixes.size(); i++)
	{
		if (gpu_suffixes[i] != cpu_output.suffixes[i])
		{
			cout << "Bad char" << endl;
			return false;
		}
	}

	thrust::host_vector<int> gpu_positions(gpu_output.positions);
	for (size_t i = 0; i < gpu_positions.size(); i++)
		if (gpu_positions[i] != cpu_output.positions[i])
		{
			cout << "Bad position" << endl;
			return false;
		}

	return true;
}

bool test_book(const char* filename)
{
	thrust::host_vector<int> positions;
	thrust::host_vector<uchar> words_chars;

	read_file(filename, positions, words_chars);

	const thrust::device_vector<uchar> words_device(words_chars);
	thrust::device_vector<int> positions_device(positions);

	thrust::device_vector<int> sorted_positions;

	get_sorted_positions(positions_device, words_device, sorted_positions);
	const bool sorting_result = test_string_sorting(sorted_positions, words_chars);

	sorting_output_gpu output;
	create_output(words_device, sorted_positions, output);
	const bool output_result = test_output(words_chars, output);

	if (!output_result || !sorting_result)
	{
		cout << "Fail testing book " << filename << endl;
		return false;
	}

	cout << "Win" << endl;

	return true;
}

void generate_random_strings(thrust::host_vector<uchar>& words, thrust::host_vector<int>& positions)
{
	const auto avg_len = (RANDSTRMAXLEN + RANDSTRMINLEN) / 2;
	const auto diff_len = RANDSTRMAXLEN - RANDSTRMINLEN + 1;
	const size_t chars_count = RANDSTRCOUNT * avg_len * 11 / 10;
	int words_index = 0;
	words.resize(chars_count);
	positions.resize(RANDSTRCOUNT);
	for (size_t i = 0; i < RANDSTRCOUNT; i++)
	{
		positions[i] = words_index;
		const int strlen = rand() % diff_len + RANDSTRMINLEN;
		for (int j = 0; j < strlen; j++)
			words[words_index++] = RANDCHARSET[rand() % RANDCHARSCOUNT];

		words[words_index++] = BREAKCHAR;
	}

	for (int i = 0; i < RANDSTRMAXLEN; i++)
		words[words_index++] = BREAKCHAR;

	words.resize(words_index);
}

bool test_random_strings()
{
	thrust::host_vector<uchar> words;
	thrust::host_vector<int> positions;
	generate_random_strings(words, positions);

	const thrust::device_vector<uchar> words_device(words);
	thrust::device_vector<int> positions_device(positions);
	thrust::device_vector<int> positions_device2(positions);

	thrust::device_vector<int> sorted_positions;
	std::cout << measure::execution(get_sorted_positions, positions_device, words_device, sorted_positions) << " Sorting random strings" << endl;

	std::cout << measure::execution(sort_positions_thrust, positions_device2, words_device) << " Sorting random strings thrust" << endl;

	const bool sorting_result = test_string_sorting(sorted_positions, words);

	sorting_output_gpu output;
	create_output(words_device, sorted_positions, output);
	const bool output_result = test_output(words, output);

	return output_result && sorting_result;
}

using namespace thrust;
void get_gpu_result(const host_vector<int>& positions_dictionary, const host_vector<uchar>& words_dictionary,
	const host_vector<int>& positions_book_host, const host_vector<uchar>& words_book, device_vector<bool>& result)
{
	device_vector<int> positions_book;
	device_vector<unsigned char> words;
	device_vector<int> sorted_positions;

	prepare_for_search(positions_dictionary, words_dictionary, positions_book_host, words_book,
		positions_book, words, sorted_positions);

	std::cout << measure::execution_gpu(find_if_strings_exist, positions_book, sorted_positions, words, result) << "gpu microseconds taken finding result" << std::endl;
}

void fill_result_vec(vector<bool>& result, const std::vector<std::string>& strings_book, const unordered_set<string>& dictionary)
{
	result.resize(strings_book.size());

	const auto end = dictionary.end();
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto val = dictionary.find(strings_book[i]);
		result[i] = val != end;
	}
}

void get_cpu_result(const host_vector<uchar>& words_dictionary, const host_vector<uchar>& words_book,
	const host_vector<int>& positions_book, vector<bool>& result)
{
	auto strings_dictionary = get_sorted_unique_cpu_words(words_dictionary);
	vector<string> strings_book(positions_book.size());
	vector<uchar> chars;

	for (size_t i = 0; i < strings_book.size(); i++)
	{
		const auto position = positions_book[i];
		int index_in_word = 0;
		while (true)
		{
			const auto c = words_book[position + index_in_word];
			if (c != BREAKCHAR)
			{
				chars.push_back(c);
				index_in_word++;
			}
			else
			{
				strings_book[i] = string(chars.begin(), chars.end());
				chars.clear();
				break;
			}
		}
	}

	const unordered_set<string> dictionary(strings_dictionary.begin(), strings_dictionary.end());

	std::cout << measure::execution(fill_result_vec, result, strings_book, dictionary) << "cpu microseconds taken finding result" << std::endl;
}

bool test_array_searching_book(const char* dictionary_filename, const char* book_filename)
{
	host_vector<int> positions_dictionary_host;
	host_vector<uchar> words_dictionary_host;
	read_file(dictionary_filename, positions_dictionary_host, words_dictionary_host);

	host_vector<int> positions_book_host;
	host_vector<uchar> words_book_host;
	read_file(book_filename, positions_book_host, words_book_host);

	device_vector<bool> gpu_result;
	vector<bool> cpu_result;

	std::cout << measure::execution(get_gpu_result, positions_dictionary_host, words_dictionary_host,
		positions_book_host, words_book_host, gpu_result) <<
		"gpu microseconds total taken finding result" << std::endl;
	std::cout << measure::execution(get_cpu_result, words_dictionary_host, words_book_host, positions_book_host, cpu_result) <<
		"cpu microseconds total taken finding result" << std::endl;

	if (gpu_result.size() != cpu_result.size())
	{
		cout << "fail searching result";
		return false;
	}
	auto strings_book = get_sorted_cpu_words(words_book_host);

	auto vec_result = from_vector_dev(gpu_result);
	for (size_t i = 0; i < cpu_result.size(); i++)
	{
		if (cpu_result[i] != vec_result[i])
		{
			auto s = strings_book[i];
			cout << "fail searching result";
			return false;
		}
	}

	cout << "array searching win" << endl;
	return true;
}
