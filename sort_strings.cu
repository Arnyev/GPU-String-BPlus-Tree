#include "gpu_helper.cuh"
#include <helper_cuda.h>
#include <sort_strings.cuh>
#include "thrust/sort.h"
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>

using namespace thrust;

__global__ void reposition_strings_d(unsigned char* d_word_array_in, unsigned char* d_word_array_out,
	int* d_position_in, int* d_position_out, const int word_count)
{
	const int thread_num = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread_num >= word_count)
		return;

	const int position_in = d_position_in[thread_num];
	const int position_out = d_position_out[thread_num];

	int i = 0;
	char c;
	do
	{
		c = d_word_array_in[position_in + i];
		d_word_array_out[position_out + i] = c;
		i++;
	} while (c != BREAKCHAR);
}

__global__ void flag_different_than_last_d(const ullong* keys, int* flags, const size_t word_count)
{
	const int thread_num = GetGlobalId();
	if (thread_num >= word_count)
		return;

	flags[thread_num] = thread_num == 0 || keys[thread_num] != keys[thread_num - 1];
}

__global__ void create_hashes_with_seg_d(const uchar* words, const int* word_positions, const int* segments, ullong* keys,
	const int offset, const int chars_to_hash, const int seg_shift, const size_t word_count)
{
	const auto thread_num = GetGlobalId();
	if (thread_num >= word_count)
		return;

	const auto position = word_positions[thread_num] + offset;

	keys[thread_num] = static_cast<ullong>(segments[thread_num]) << seg_shift | get_hash(words, chars_to_hash, position);
}

__global__ void mark_singletons_d(const ullong* keys, int* flags, const int* destinations, int* output,
	const int* positions, const size_t word_count)
{
	const int thread_num = GetGlobalId();
	if (thread_num >= word_count)
		return;

	const auto key = keys[thread_num];
	const auto position = positions[thread_num];
	const auto finished = (key & 1ULL) == 0ULL;
	const auto index_output = destinations[thread_num];

	if (thread_num == 0)
	{
		if (finished || key != keys[thread_num + 1])
		{
			output[index_output] = position;
			flags[thread_num] = 0;
		}
		else
			flags[thread_num] = 1;

		return;
	}

	const auto key_last = keys[thread_num - 1];

	if (thread_num == word_count - 1)
	{
		if (key != key_last)
		{
			output[index_output] = position;
			flags[thread_num] = 0;
		}
		else if (finished)
		{
			output[index_output] = -1;
			flags[thread_num] = 0;
		}
		else
			flags[thread_num] = 1;

		return;
	}

	const auto key_next = keys[thread_num + 1];

	if (key != key_last && (finished || key != key_next))
	{
		output[index_output] = position;
		flags[thread_num] = 0;
	}
	else if (key == key_last && finished)
	{
		output[index_output] = -1;
		flags[thread_num] = 0;
	}
	else
		flags[thread_num] = 1;
}

__global__ void compute_postfix_lengths_d(uchar* words, int* positions, const int word_count, int* lengths)
{
	const int thread_num = GetGlobalId();
	if (thread_num >= word_count)
		return;

	int my_position = positions[thread_num];
	if (my_position == -1)
		return;

	int length = 0;
	uchar c;
	for (int i = 1; i < CHARSTOHASH; i++)
	{
		c = words[my_position + i];
		if (c == BREAKCHAR)
			return;
	}

	my_position = my_position + CHARSTOHASH;
	while (true)
	{
		c = words[my_position];

		if (c == BREAKCHAR)
			break;
		my_position++;
		length++;
	}

	lengths[thread_num] = length + 1;
}

__global__ void copy_suffixes_d(const uchar* words, const int* positions, const size_t word_count, uchar* suffixes, const int* suffix_positions)
{
	const auto thread_num = GetGlobalId();
	if (thread_num >= word_count)
		return;

	const int suffix_pos = suffix_positions[thread_num];
	const int len = suffix_positions[thread_num + 1] - suffix_pos;
	if (len == 0)
		return;

	const int position = positions[thread_num] + CHARSTOHASH;

	for (int i = 0; i < len; i++)
		suffixes[suffix_pos + i] = words[position + i];
}

void copy_suffixes(const device_vector<uchar>& words, const device_vector<int>& sorted_positions, const size_t word_count,
                   const device_vector<int> suffix_positions, device_vector<uchar>& suffixes)
{
	STARTKERNEL(copy_suffixes_d, "Copying suffixes", word_count, words.data().get(), sorted_positions.data().get(),
		word_count,suffixes.data().get(), suffix_positions.data().get());
}

void flags_different_than_last(const device_vector<ullong>& keys, device_vector<int>& flags)
{
	STARTKERNEL(flag_different_than_last_d, "Flags different than last", keys.size(), keys.data().get(), flags.
		data().get(), keys.size());
}

void create_hashes_with_seg(const device_vector<int>& positions, const device_vector<uchar>& chars,
	device_vector<ullong>& keys, const device_vector<int>& segments, const int offset, const int segment_size, const int seg_chars)
{
	STARTKERNEL(create_hashes_with_seg_d, "Create hashes", positions.size(), chars.data().get(), positions.data().
		get(), segments.data().get(), keys.data().get(), offset, CHARSTOHASH - seg_chars, KEYBITS - segment_size,
		positions.size());
}

void mark_singletons(const device_vector<int>& positions, const device_vector<ullong>& keys,
	const device_vector<int>& destinations, device_vector<int>& flags, device_vector<int>& output)
{
	STARTKERNEL(mark_singletons_d, "Marking singletons", positions.size(), keys.data().get(), flags.data().get(),
		destinations.data().get(), output.data().get(), positions.data().get(), positions.size());
}

void remove_handled(device_vector<int>& positions, device_vector<ullong>& keys, device_vector<int>& destinations,
	device_vector<int>& helper)
{
	const auto iter_start = make_zip_iterator(make_tuple(keys.begin(), positions.begin(), destinations.begin()));
	const auto iter_end = make_zip_iterator(make_tuple(keys.end(), positions.end(), destinations.end()));

	const auto new_end = remove_if(iter_start, iter_end, helper.begin(), equal_to_val<uchar, 0>());

	const auto current_count = new_end - iter_start;
	positions.resize(current_count);
	keys.resize(current_count);
	destinations.resize(current_count);
	helper.resize(current_count);
}

int compute_segment_size(const device_vector<int>& segments)
{
	int max_segment = segments.back();
	int segment_size;

	if (max_segment == 0)
		segment_size = 0;
	else
	{
		segment_size = 32;
		const int flag = 1 << (sizeof(int) * 8 - 1);
		while ((max_segment&flag) == 0)
		{
			max_segment <<= 1;
			segment_size--;
		}
	}

	return segment_size;
}

void sort_positions(device_vector<int>& positions, device_vector<ullong>& keys)
{
	sort_by_key(keys.begin(), keys.end(), positions.begin());
}

void get_sorted_positions(device_vector<int>& positions, const device_vector<uchar>& chars, device_vector<int>& output)
{
	device_vector<ullong> keys(positions.size());
	device_vector<int> destinations(positions.size());
	device_vector<int> helper(positions.size());
	output.reserve(positions.size() + 1);
	output.resize(positions.size());

	sequence(destinations.begin(), destinations.end());

	int offset = 0;
	int segment_size = 0;

	while (true)
	{
		const auto seg_chars = static_cast<int>(ceil(static_cast<double>(segment_size) / CHARBITS));
		const auto hashing_time = measure::execution_gpu(create_hashes_with_seg, positions, chars, keys, helper, offset, segment_size, seg_chars);

		if (WRITETIME)
			std::cout << hashing_time << " microseconds taken creating hashes" << std::endl;

		offset += CHARSTOHASH - seg_chars;

		const auto sorting_time = measure::execution_gpu(sort_positions, positions, keys);

		if (WRITETIME)
			std::cout << sorting_time << " microseconds taken sorting" << std::endl;

		mark_singletons(positions, keys, destinations, helper, output);

		remove_handled(positions, keys, destinations, helper);
		if (positions.empty())
			break;

		flags_different_than_last(keys, helper);

		inclusive_scan(helper.begin(), helper.end(), helper.begin());
		segment_size = compute_segment_size(helper);
	}
}

void sort_positions_thrust(device_vector<int>& positions, const device_vector<uchar>& chars)
{
	sort(positions.begin(), positions.end(), less_than_string(chars.data().get()));
}

void create_output(const device_vector<uchar>& words, device_vector<int>& sorted_positions, sorting_output_gpu& result)
{
	const auto positions_end = remove_if(sorted_positions.begin(), sorted_positions.end(), equal_to_val<int, -1>());

	const auto word_count = static_cast<int>(positions_end - sorted_positions.begin());
	sorted_positions.resize(word_count + 1);

	result.positions.resize(word_count + 1);

	const compute_postfix_length_functor postfix_functor(words.data().get());
	transform_exclusive_scan(sorted_positions.begin(), sorted_positions.end(), result.positions.begin(), postfix_functor, 0, thrust::plus<int>());

	const int output_size = result.positions.back();

	result.suffixes.resize(output_size);

	copy_suffixes(words, sorted_positions, word_count, result.positions, result.suffixes);

	result.hashes.resize(word_count);
	transform(sorted_positions.begin(), sorted_positions.begin()+word_count, result.hashes.begin(), hash_functor(words.data().get()));

	const auto hashes_end = unique_by_key(result.hashes.begin(), result.hashes.end(), result.positions.begin());
	const auto hashes_count = hashes_end.first - result.hashes.begin();
	result.hashes.resize(hashes_count);
	result.positions.resize(hashes_count);
}

void find_if_strings_exist(const device_vector<int>& values_positions, const device_vector<int>& input_positions,
	const device_vector<uchar>& words, device_vector<bool>& result)
{
	result.resize(values_positions.size());
	const less_than_string func(words.data().get());

	binary_search(input_positions.begin(), input_positions.end(), values_positions.begin(), values_positions.end(), result.begin(), func);
}

void prepare_for_search(const host_vector<int>& positions_dictionary_host, const host_vector<uchar>& words_dictionary_host,
	const host_vector<int>& positions_book_host, const host_vector<uchar>& words_book_host, device_vector<int>& positions_book,
	device_vector<unsigned char>& words, device_vector<int>& sorted_positions)
{
	device_vector<int> positions_dictionary(positions_dictionary_host);

	positions_book = positions_book_host;

	using namespace thrust::placeholders;
	transform(positions_book.begin(), positions_book.end(), positions_book.begin(), _1 + words_dictionary_host.size());

	words = device_vector<uchar>(words_dictionary_host.size() + words_book_host.size() + CHARSTOHASH);
	copy(words_dictionary_host.begin(), words_dictionary_host.end(), words.begin());
	copy(words_book_host.begin(), words_book_host.end(), words.begin() + words_dictionary_host.size());

	get_sorted_positions(positions_dictionary, words, sorted_positions);
	const auto new_end = remove_if(sorted_positions.begin(), sorted_positions.begin() + sorted_positions.size(), equal_to_val<int, -1>());

	const auto dict_count = new_end - sorted_positions.begin();
	sorted_positions.resize(dict_count);
}
