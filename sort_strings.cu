#include "gpu_helper.cuh"
#include "thrust/sort.h"
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include "sort_helpers.cuh"
#include "helpers.h"

using namespace thrust;

__global__ void flag_different_than_last_d(const ullong* keys, int* flags, const size_t word_count)
{
	const int thread_num = GetGlobalId();
	if (thread_num >= word_count)
		return;

	flags[thread_num] = thread_num == 0 || keys[thread_num] != keys[thread_num - 1];
}

__global__ void create_hashes_with_seg_d(const char* words, const int* word_positions, const int* segments, ullong* keys,
	const int offset, const int chars_to_hash, const int seg_shift, const size_t word_count)
{
	const auto thread_num = GetGlobalId();
	if (thread_num >= word_count)
		return;

	const auto position = word_positions[thread_num] + offset;

	keys[thread_num] = static_cast<ullong>(segments[thread_num]) << seg_shift | get_hash<ullong>(words, position, chars_to_hash);
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

void flags_different_than_last(const device_vector<ullong>& keys, device_vector<int>& flags)
{
	STARTKERNEL(flag_different_than_last_d, "Flags different than last", keys.size(), keys.data().get(), flags.
		data().get(), keys.size());
}

void create_hashes_with_seg(const device_vector<int>& positions, const device_vector<char>& chars,
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

	const auto new_end = remove_if(iter_start, iter_end, helper.begin(), equal_to_val<char, 0>());

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

void get_sorted_positions(device_vector<int>& positions, const device_vector<char>& chars, device_vector<int>& output)
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

void get_sorted_positions_no_duplicates(device_vector<int>& positions, const device_vector<char>& chars, device_vector<int>& output)
{
	get_sorted_positions(positions, chars, output);

	const auto positions_end = remove_if(output.begin(), output.end(), equal_to_val<int, -1>());
	output.resize(positions_end - output.begin());
}

void sort_positions_thrust(device_vector<int>& positions, const device_vector<char>& chars)
{
	sort(positions.begin(), positions.end(), less_than_string(chars.data().get()));
}
