#pragma once

#include <thrust/device_ptr.h>

void sort_keys_and_positions(int* d_positions, thrust::device_ptr<unsigned long long> keys, int current_count);

void mark_singletons(int* d_positions, thrust::device_ptr<unsigned long long> keys,
	thrust::device_ptr<int> destinations, thrust::device_ptr<int> helper,
	thrust::device_ptr<int> output, int current_count);

void get_segments(thrust::device_ptr<int> helper, int current_count);

void create_hashes_with_seg(int* d_positions, unsigned char* d_chars, thrust::device_ptr<unsigned long long> keys,
	thrust::device_ptr<int> helper, int offset, int segment_size, int current_count,
	int seg_chars);

void remove_handled_update_count(thrust::device_ptr<int> positions, thrust::device_ptr<ullong> keys,
	thrust::device_ptr<int> destinations, thrust::device_ptr<int> helper, int& current_count);

void create_consecutive_numbers(int word_count, thrust::device_ptr<int> destinations);

void flags_different_than_last(ullong* d_keys, int* d_flags, int current_count);

thrust::device_ptr<int> remove_duplicates(int* d_sorted_positions, int word_count, thrust::device_ptr<int> sorted_positions);

void create_hashes(uchar* d_word_array, thrust::device_ptr<int> sorted_positions, thrust::device_ptr<int> positions_end,
	thrust::device_ptr<ullong> hashes);

void get_suffix_positions(unsigned char* d_word_array, thrust::device_ptr<int> sorted_positions,
	thrust::device_ptr<int> positions_end, thrust::device_ptr<int> suffix_positions);

void copy_suffixes(unsigned char* d_word_array, int* d_sorted_positions, int word_count,
	thrust::device_ptr<int> suffix_positions, thrust::device_ptr<unsigned char> suffixes);

thrust::device_ptr<unsigned long long> get_unique_hashes(int word_count, thrust::device_ptr<int> suffix_positions,
	thrust::device_ptr<unsigned long long> hashes);
