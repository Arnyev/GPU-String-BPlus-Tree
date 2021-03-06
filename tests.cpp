#include <algorithm>
#include <string>
#include <unordered_set>
#include "functions.h"
#include <numeric>
#include <helper_cuda.h>
#include <cctype>
#include <regex>
#include "dictionary_reader.h"
#include "book_reader.h"
#include "csv_logger.h"

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
	vector<string> strings_book;
	create_strings(words_book, positions_book, strings_book);

	const unordered_set<string> dictionary(strings_dictionary.begin(), strings_dictionary.end());

	std::cout << measure::execution(fill_result_vec, result, strings_book, dictionary) << "cpu microseconds taken finding result" << std::endl;
}

void test_array_searching_book(dictionary_reader &dictReader, book_reader &bookReader, csv_logger &logger)
{
	float preprocTime, execTime, postTime;
	std::vector<int> positions_dict_std_vector;
	std::vector<uchar> chars_dict_std_vector;
	std::tie(positions_dict_std_vector, chars_dict_std_vector) = dictReader.get_words<uchar>();
	host_vector<int> positions_dictionary_host(positions_dict_std_vector);
	host_vector<uchar> words_dictionary_host(chars_dict_std_vector);

	std::vector<int> positions_book_std_vector;
	std::vector<uchar> chars_book_std_vector;

	auto start = std::chrono::steady_clock::now();
	auto book_result = bookReader.get_words(1);
	host_vector<int> positions_book_host(std::get<1>(book_result));
	host_vector<uchar> words_book_host(std::get<0>(book_result));
	preprocTime = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();

	device_vector<bool> gpu_result;
	vector<bool> cpu_result;

	device_vector<int> positions_book;
	device_vector<unsigned char> words;
	device_vector<int> sorted_positions;

	const auto build_time = measure::execution_gpu(prepare_for_search, positions_dictionary_host, words_dictionary_host,
		positions_book_host, words_book_host, positions_book, words, sorted_positions);

	execTime = measure::execution_gpu(find_if_strings_exist, positions_book, sorted_positions, words, gpu_result) / 1000;

	//get_cpu_result(words_dictionary_host, words_book_host, positions_book_host, cpu_result);

	//if (gpu_result.size() != cpu_result.size())
	//	return false;

	start = std::chrono::steady_clock::now();
	auto vec_result = from_vector_dev(gpu_result);
	postTime = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
	//for (size_t i = 0; i < cpu_result.size(); i++)
	//	if (cpu_result[i] != vec_result[i])
	//		return false;

	const int existing = std::count(vec_result.begin(), vec_result.end(), true);

	const auto percent_existing = static_cast<double>(existing) / bookReader.words_count();

	logger.append("Thrust binary search", build_time / 1000, preprocTime, execTime, postTime, dictReader.words_count(), bookReader.words_count(), percent_existing);
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

	device_vector<int> positions_book;
	device_vector<unsigned char> words;
	device_vector<int> sorted_positions;

	const auto build_time = measure::execution_gpu(prepare_for_search, positions_dictionary_host, words_dictionary_host,
		positions_book_host, words_book_host, positions_book, words, sorted_positions);

	const auto time_finding = measure::execution_gpu(find_if_strings_exist, positions_book, sorted_positions, words, gpu_result);

	get_cpu_result(words_dictionary_host, words_book_host, positions_book_host, cpu_result);

	if (gpu_result.size() != cpu_result.size())
		return false;

	auto vec_result = from_vector_dev(gpu_result);
	for (size_t i = 0; i < cpu_result.size(); i++)
		if (cpu_result[i] != vec_result[i])
			return false;

	int existing = 0;
	for (const auto res : cpu_result)
		if (res)
			existing++;

	const auto percent_existing = static_cast<double>(existing) / cpu_result.size();

	append_to_csv("Thrust binary search", build_time / 1000, time_finding / 1000, sorted_positions.size(), positions_book_host.size(), percent_existing);

	return true;
}

