#pragma once

#include <parameters.h>

template<class T, T N>
struct equal_to_val : thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T x) const { return x == N; }
};

struct less_than_string : thrust::binary_function<int, int, bool>
{
	const uchar * words;
	explicit less_than_string(const uchar* words) : words(words) {	}

	__host__ __device__ bool operator()(const int x, const int y) const
	{
		int i = 0;
		while (true)
		{
			const auto c1 = words[x + i];
			const auto c2 = words[y + i];
			if (c1 < c2)
				return true;
			if (c2 < c1)
				return false;
			if (c1 == 0)
				return false;
			++i;
		}
	}
};

__device__ __host__ __inline__ ullong get_hash(const uchar* words, const int chars_to_hash, const int my_position)
{
	uchar last_bit = 1;
	uchar char_mask = CHARMASK;

	ullong hash = 0;

	for (int i = 0; i < chars_to_hash; i++)
	{
		const unsigned char c = words[i + my_position];
		if (c == BREAKCHAR)
		{
			char_mask = 0;
			last_bit = 0;
		}
		hash *= ALPHABETSIZE;
		hash += c & char_mask;
	}
	if (words[chars_to_hash + my_position] == BREAKCHAR)
		last_bit = 0;

	return hash << 1 | last_bit;
}

struct hash_functor : thrust::unary_function<int, ullong>
{
	const uchar* words;

	explicit hash_functor(const uchar* words) : words(words) {	}

	__host__ __device__ ullong operator()(const int position) const
	{
		return get_hash(words, CHARSTOHASH, position);
	}
};

struct compute_postfix_length_functor : thrust::unary_function<int, int>
{
	const uchar* words;

	explicit compute_postfix_length_functor(const uchar* words) : words(words) {}

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
				return length + 1;

			my_position++;
			length++;
		}
	}
};