#pragma once

#include <thrust/device_vector.h>

std::vector<char> read_file_to_buffer(const char* filepath);
void read_file(const char* filepath, thrust::host_vector<int>& positions, thrust::host_vector<char>& words);

void copy_suffixes(const thrust::device_vector<char>& words, const thrust::device_vector<int>& sorted_positions, size_t word_count,
	const thrust::device_vector<int>& suffix_positions, thrust::device_vector<char>& suffixes);

void get_strings(const thrust::host_vector<char>& words, const thrust::host_vector<int>& positions, std::vector<std::string>& strings);
void get_sorted_cpu_strings(const thrust::host_vector<char>& words, const thrust::host_vector<int>& positions, std::vector<std::string>& strings);
void get_sorted_unique_cpu_strings(const thrust::host_vector<char>& words, const thrust::host_vector<int>& positions, std::vector<std::string>& strings);

void get_sorted_positions(thrust::device_vector<int>& positions, const thrust::device_vector<char>& chars, thrust::device_vector<int>& output);
void get_sorted_positions_no_duplicates(thrust::device_vector<int>& positions, const thrust::device_vector<char>& chars, thrust::device_vector<int>& output);
void sort_positions_thrust(thrust::device_vector<int>& positions, const thrust::device_vector<char>& chars);
void get_strings(const thrust::host_vector<char>& words_book, const thrust::host_vector<int>& positions_book, std::vector<std::string>& strings_book);
void generate_random_strings(thrust::host_vector<char>& words, thrust::host_vector<int>& positions);

void prepare_for_search(
	const thrust::host_vector<int>& positions_dictionary_host,
	const thrust::host_vector<char>& words_dictionary_host,
	const thrust::host_vector<int>& positions_book_host,
	const thrust::host_vector<char>& words_book_host,
	thrust::device_vector<int>& positions_book,
	thrust::device_vector<char>& words,
	thrust::device_vector<int>& positions_dictionary);

void find_if_strings_exist(
	const thrust::device_vector<int>& values_positions,
	const thrust::device_vector<int>& input_positions,
	const thrust::device_vector<char>& words,
	thrust::device_vector<bool>& result);

void append_to_csv(const char* algorithm, float build_time, float execution_time, size_t dict_size,
	size_t input_size, double existing_percentage);

void search_cpu(const thrust::host_vector<char>& words_dictionary, const thrust::host_vector<int>& positions_dictionary,
	const thrust::host_vector<char>& words_book, const thrust::host_vector<int>& positions_book,
	std::vector<bool>& result);

bool test_random_strings();
bool test_array_searching_book(const char* dictionary_filename, const char* book_filename);
bool test_book(const char* filename);
bool test_output(const thrust::host_vector<char>& words, const thrust::host_vector<int>& positions_host,
	const thrust::device_vector<char>& words_device, thrust::device_vector<int>& sorted_positions);
void test_tree();
