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

sorting_output create_output(unsigned char* d_wordArray, int char_count, int* d_sortedPositions, int word_count);
int* get_sorted_positions(unsigned char* h_wordArray, int* h_wordPositions, int* h_wordLengths, int wordCount,
	size_t wordArraySize);
bool test_string_sorting(int * d_positions, const int word_count, unsigned char * h_word_array, const size_t word_array_size);
bool test_output(unsigned char* d_word_array, int chars_input_count, sorting_output output);
