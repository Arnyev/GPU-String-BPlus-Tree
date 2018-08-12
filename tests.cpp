#include "parameters.h"
#include <algorithm>
#include "functions.h"
#include "helpers.h"

using namespace std;

vector<string> get_sorted_gpu_words(const thrust::device_vector<int>& sorted_positions, const thrust::host_vector<char> & words)
{
	thrust::host_vector<int> positions(sorted_positions.size());
	thrust::copy(sorted_positions.begin(), sorted_positions.end(), positions.begin());
	vector<char> word;
	vector<string> result;

	for (size_t i = 0; i < sorted_positions.size(); i++)
	{
		const int position = positions[i];
		int index_in_word = 0;
		while (true)
		{
			const char c = words[position + index_in_word];
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

bool test_string_sorting(const thrust::device_vector<int>& sorted_positions, const thrust::host_vector<char> & words, const thrust::host_vector<int> & positions)
{
	auto words_gpu = get_sorted_gpu_words(sorted_positions, words);
	vector<string> words_cpu;
	get_sorted_unique_cpu_strings(words, positions, words_cpu);

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

using namespace thrust;

bool test_book(const char* filename)
{
	host_vector<int> positions;
	host_vector<char> words;

	read_file(filename, positions, words);

	const device_vector<char> words_device(words);
	device_vector<int> positions_device(positions);
	device_vector<int> sorted_positions;

	get_sorted_positions_no_duplicates(positions_device, words_device, sorted_positions);
	const bool sorting_result = test_string_sorting(sorted_positions, words, positions);

	const bool output_result = test_output(words, positions, words_device, sorted_positions);

	if (!output_result || !sorting_result)
	{
		cout << "Fail testing book " << filename << endl;
		return false;
	}

	cout << "Win" << endl;

	return true;
}

void sort_strings_std(std::vector<std::string> strings)
{
	std::sort(strings.begin(), strings.end());
}

bool test_random_strings()
{
	host_vector<char> words;
	host_vector<int> positions;
	generate_random_strings(words, positions);

	const device_vector<char> words_device(words);
	device_vector<int> positions_device(positions);
	device_vector<int> positions_device2(positions);

	device_vector<int> sorted_positions;
	std::vector<std::string> strings;
	get_strings(words, positions, strings);

	std::cout << measure::execution_gpu(get_sorted_positions_no_duplicates, positions_device, words_device, sorted_positions) << " Sorting random strings gpu" << endl;

	std::cout << measure::execution_gpu(sort_positions_thrust, positions_device2, words_device) << " Sorting random strings thrust" << endl;

	std::cout << measure::execution(sort_strings_std, strings) << " Sorting random strings std" << endl;

	const bool sorting_result = test_string_sorting(sorted_positions, words, positions);

	const bool output_result = test_output(words, positions, words_device, sorted_positions);

	return output_result && sorting_result;
}

bool test_array_searching_book(const char* dictionary_filename, const char* book_filename)
{
	host_vector<int> positions_dictionary_host;
	host_vector<char> words_dictionary_host;
	read_file(dictionary_filename, positions_dictionary_host, words_dictionary_host);

	host_vector<int> positions_book_host;
	host_vector<char> words_book_host;
	read_file(book_filename, positions_book_host, words_book_host);

	device_vector<bool> gpu_result;
	vector<bool> cpu_result;

	device_vector<int> positions_book;
	device_vector<char> words;
	device_vector<int> sorted_positions;

	const auto build_time = measure::execution_gpu(prepare_for_search, positions_dictionary_host, words_dictionary_host,
		positions_book_host, words_book_host, positions_book, words, sorted_positions);

	const auto time_finding = measure::execution_gpu(find_if_strings_exist, positions_book, sorted_positions, words, gpu_result);

	search_cpu(words_dictionary_host, positions_dictionary_host, words_book_host, positions_book_host, cpu_result);

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
