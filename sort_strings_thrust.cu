#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "parameters.h"
#include "functions.h"
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/transform_scan.h>
#include <gpu_helper.cuh>
#include "sort_helpers.cuh"

using namespace thrust;
using namespace std;

template<class T>
struct equal_to_minus_one : thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T x) const { return x == -1; }
};

template<class T>
struct equal_to_zero : thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T x) const { return x == 0; }
};

struct hash_functor : thrust::unary_function<int, ullong>
{
	uchar* words;

	explicit hash_functor(uchar* words) : words(words) {	}

	__host__ __device__ ullong operator()(const int position) const
	{
		if (position == -1)
			return 0ULL;

		return get_hash(words, CHARSTOHASH, position);
	}
};

struct compute_postfix_length_functor : thrust::unary_function<int, int>
{
	uchar* words;

	__device__  int operator()(int my_position) const
	{
		if (my_position == -1)
			return 0;

		int length = 0;
		uchar c;
		for (int i = 1; i < CHARSTOHASH; i++)
		{
			c = words[my_position + i];
			if (c == BREAKCHAR)
				return 0;
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

		return length + 1;
	}
};

void create_hashes(uchar* d_word_array, const device_ptr<int> sorted_positions, const device_ptr<int> positions_end,
	const device_ptr<ullong> hashes)
{
	MEASURETIME(transform(sorted_positions, positions_end, hashes, hash_functor(d_word_array)), "Hashes simple");
}

device_ptr<int> remove_duplicates(int* d_sorted_positions, const int word_count, const device_ptr<int> sorted_positions)
{
	device_ptr<int> positions_end;
	MEASURETIME(positions_end = remove_if(sorted_positions, device_ptr<int>(d_sorted_positions + word_count), equal_to_minus_one<int>()), "Remove duplicates");
	return positions_end;
}

void get_suffix_positions(unsigned char* d_word_array, const device_ptr<int> sorted_positions, const device_ptr<int> positions_end, const device_ptr<int> suffix_positions)
{
	compute_postfix_length_functor postfix_functor;
	postfix_functor.words = d_word_array;
	MEASURETIME(transform_exclusive_scan(sorted_positions, positions_end + 1, suffix_positions, postfix_functor, 0, thrust::plus<int>()), "Getting suffix positions");
}

device_ptr<unsigned long long> get_unique_hashes(const int word_count, const device_ptr<int> suffix_positions,
	const device_ptr<unsigned long long> hashes)
{
	thrust::pair<device_ptr<unsigned long long>, device_ptr<int>> hashes_end;
	MEASURETIME(hashes_end = unique_by_key(hashes, hashes + word_count, suffix_positions), "Getting unique hashes");

	return hashes_end.first;
}

void sort_keys_and_positions(int* d_positions, device_ptr<unsigned long long> keys, int current_count)
{
	MEASURETIME(sort_by_key(keys, keys + current_count, device_ptr<int>(d_positions)), "Sorting ");
}

void get_segments(const device_ptr<int> helper, int current_count)
{
	MEASURETIME(inclusive_scan(helper, helper + current_count, helper), "Inclusive scan");
}

void remove_handled_update_count(const device_ptr<int> positions, const device_ptr<ullong> keys,
	const device_ptr<int> destinations, const device_ptr<int> helper, int& current_count)
{
	const auto iter_start = make_zip_iterator(thrust::make_tuple(keys, positions, destinations));
	const auto iter_end = make_zip_iterator(
		thrust::make_tuple(keys + current_count, positions + current_count, destinations + current_count));

	zip_iterator<thrust::tuple<device_ptr<unsigned long long>, device_ptr<int>, device_ptr<int>>> new_end;

	MEASURETIME(new_end = remove_if(iter_start, iter_end, helper, equal_to_zero<uchar>()), "Remove handled");
	current_count = new_end - iter_start;
}
