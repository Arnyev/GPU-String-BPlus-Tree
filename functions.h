#pragma once

#include "parameters.h"
#include "dictionary_reader.h"
#include "book_reader.h"
#include "csv_logger.h"

std::vector<char> read_file_to_buffer(const char* filepath);
void read_file(const char* filepath, thrust::host_vector<int>& positions, thrust::host_vector<uchar>& words);
void create_output(const thrust::device_vector<uchar>& words, thrust::device_vector<int>& sorted_positions, sorting_output_gpu& output);
void get_sorted_positions(thrust::device_vector<int>& positions, const thrust::device_vector<uchar>& chars, thrust::device_vector<int>& output);
void sort_positions_thrust(thrust::device_vector<int>& positions, const thrust::device_vector<uchar>& chars);
bool test_random_strings();
bool test_array_searching_book(const char* dictionary_filename, const char* book_filename);
void test_array_searching_book(dictionary_reader &dictReader, book_reader &bookReader, csv_logger &logger);
bool test_book(const char* filename);
void test_gpu_tree_vectors(const char* dictionary_filename, const char* book_filename);
void create_strings(const thrust::host_vector<uchar>& words_book, const thrust::host_vector<int>& positions_book, std::vector<std::string>& strings_book);
void generate_random_strings(thrust::host_vector<uchar>& words, thrust::host_vector<int>& positions);

std::vector<std::string> get_sorted_unique_cpu_words(const thrust::host_vector<uchar> & words_chars);
std::vector<std::string> get_sorted_cpu_words(const thrust::host_vector<uchar>& words_chars);

void prepare_for_search(
	const thrust::host_vector<int>& positions_dictionary_host,
	const thrust::host_vector<uchar>& words_dictionary_host,
	const thrust::host_vector<int>& positions_book_host,
	const thrust::host_vector<uchar>& words_book_host,
	thrust::device_vector<int>& positions_book,
	thrust::device_vector<unsigned char>& words,
	thrust::device_vector<int>& positions_dictionary);

void find_if_strings_exist(
	const thrust::device_vector<int>& values_positions,
	const thrust::device_vector<int>& input_positions,
	const thrust::device_vector<uchar>& words,
	thrust::device_vector<bool>& result);

void append_to_csv(const char* algorithm, float build_time, float execution_time, size_t dict_size,
	size_t input_size, double existing_percentage);

void get_cpu_result(const thrust::host_vector<uchar>& words_dictionary, const thrust::host_vector<uchar>& words_book,
	const thrust::host_vector<int>& positions_book, std::vector<bool>& result);