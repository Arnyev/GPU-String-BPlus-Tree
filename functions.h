#pragma once
#include "helpers.h"

struct sorting_output
{
	ullong* hashes;
	int* positions;
	uchar* suffixes;
	int hashes_count;
	int suffixes_size;
};

sorting_output create_output(unsigned char* d_wordArray, int* d_sortedPositions, int word_count);
int* get_sorted_positions(int* d_word_positions, const int word_count, unsigned char* d_word_array);
bool test_string_sorting(int * d_sorted_positions, const int word_count, unsigned char * h_word_array, const size_t word_array_size);
bool test_output(unsigned char* h_word_array, int chars_input_count, sorting_output output);
std::vector<std::string> get_sorted_cpu_words(unsigned char* h_word_array, size_t word_array_size);
bool test_random_strings();
