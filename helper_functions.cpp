#include "parameters.h"
#include <thrust/host_vector.h>
#include <ctime>
#include <iomanip>
#include <cctype>
#include <fstream>
#include <chrono>
#include <unordered_set>
#include "bplus_tree_cpu.h"

std::vector<char> read_file_to_buffer(const char* filepath)
{
	std::ifstream file(filepath, std::ios::binary | std::ios::ate);
	const std::streamsize size = file.tellg();
	if (size <= 0)
	{
		std::cout << "Fail reading file " << filepath;
		exit(EXIT_FAILURE);
	}

	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size))
	{
		std::cout << "Fail reading file " << filepath;
		exit(EXIT_FAILURE);
	}

	return buffer;
}

void read_file(const char* filepath, thrust::host_vector<int>& positions, thrust::host_vector<char>& words)
{
	auto chars = read_file_to_buffer(filepath);

	auto c = '\0';

	size_t starting_position = 0;
	for (; !isalpha(c) && starting_position < chars.size(); starting_position++)
		c = chars[starting_position];

	starting_position--;
	positions.push_back(0);
	auto currently_on_not_alpha_seq = false;

	const char mask = TOLOWERMASK;
	for (auto i = starting_position; i < chars.size(); i++)
	{
		c = chars[i];

		if (c > 0 && std::isalpha(c))
		{
			if (currently_on_not_alpha_seq)
			{
				positions.push_back(static_cast<int>(words.size()));
				currently_on_not_alpha_seq = false;
			}

			words.push_back(c | mask);
		}
		else if (!currently_on_not_alpha_seq)
		{
			words.push_back(BREAKCHAR);
			currently_on_not_alpha_seq = true;
		}
	}

	for (int i = 0; i < CHARSTOHASH; i++)
		words.push_back(BREAKCHAR);
}

void get_strings(const thrust::host_vector<char>& words, const thrust::host_vector<int>& positions, std::vector<std::string>& strings)
{
	strings.resize(positions.size());
	std::vector<char> chars;

	for (size_t i = 0; i < strings.size(); i++)
	{
		const auto position = positions[i];
		int index_in_word = 0;
		while (true)
		{
			const auto c = words[position + index_in_word];
			if (c != BREAKCHAR)
			{
				chars.push_back(c);
				index_in_word++;
			}
			else
			{
				strings[i] = std::string(chars.begin(), chars.end());
				chars.clear();
				break;
			}
		}
	}
}

void get_sorted_cpu_strings(const thrust::host_vector<char>& words, const thrust::host_vector<int>& positions, std::vector<std::string>& strings)
{
	get_strings(words, positions, strings);
	std::sort(strings.begin(), strings.end());
}

void get_sorted_unique_cpu_strings(const thrust::host_vector<char>& words, const thrust::host_vector<int>& positions, std::vector<std::string>& strings)
{
	get_sorted_cpu_strings(words, positions, strings);
	const auto end = unique(strings.begin(), strings.end());

	strings.resize(std::distance(strings.begin(), end));
}

void append_to_csv(const char* algorithm, const float build_time, const float execution_time,
	const size_t dict_size, const size_t input_size, const double existing_percentage)
{
	int device;
	cudaGetDevice(&device);

	struct cudaDeviceProp props {};
	cudaGetDeviceProperties(&props, device);

	std::ofstream outfile;

	const auto time_point = std::chrono::system_clock::now();

	const auto time = std::chrono::system_clock::to_time_t(time_point);
	struct tm timeinfo{};
	localtime_s(&timeinfo, &time);

	outfile.open("results.csv", std::ios_base::app);
	outfile << std::put_time(&timeinfo, "%c") << ",\t\t" << algorithm << ",\t\t" << props.name << ",\t\t" << build_time << ",\t\t" <<
		execution_time << ",\t\t" << dict_size << ",\t\t" << input_size << ",\t\t" << existing_percentage << std::endl;
}

void fill_result_vec(std::vector<bool>& result, const std::vector<std::string>& strings_book, const std::unordered_set<std::string>& dictionary)
{
	result.resize(strings_book.size());

	const auto end = dictionary.end();
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto val = dictionary.find(strings_book[i]);
		result[i] = val != end;
	}
}

void search_cpu(const thrust::host_vector<char>& words_dictionary, const thrust::host_vector<int>& positions_dictionary,
                const thrust::host_vector<char>& words_book, const thrust::host_vector<int>& positions_book,
                std::vector<bool>& result)
{
	std::vector<std::string> strings_dictionary;
	get_sorted_unique_cpu_strings(words_dictionary, positions_dictionary, strings_dictionary);
	std::vector<std::string> strings_book;
	get_strings(words_book, positions_book, strings_book);

	const std::unordered_set<std::string> dictionary(strings_dictionary.begin(), strings_dictionary.end());

	fill_result_vec(result, strings_book, dictionary);
}

void generate_random_strings(thrust::host_vector<char>& words, thrust::host_vector<int>& positions)
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
