#pragma once

#include "parameters.h"
#include <type_traits>
#include <crt/host_defines.h>

#ifdef _MSC_VER
template<int SIZE>
constexpr int __msc__chars_to_hash()
{
	return -1;
}

template<>
constexpr int __msc__chars_to_hash<1>()
{
	return 1;
}

template<>
constexpr int __msc__chars_to_hash<2>()
{
	return 3;
}

template<>
constexpr int __msc__chars_to_hash<4>()
{
	return 6;
}

template<>
constexpr int __msc__chars_to_hash<8>()
{
	return 13;
}

template<int SIZE>
constexpr int __chars_to_hash = __msc__chars_to_hash<SIZE>();
#else
// Visual does not support variable template specialization
// https://developercommunity.visualstudio.com/content/problem/258652/variable-template-specialization-gives-error-in-ed.html
template<int SIZE>
constexpr int __chars_to_hash = -1;

template<>
constexpr int __chars_to_hash<1> = 1;

template<>
constexpr int __chars_to_hash<2> = 3;

template<>
constexpr int __chars_to_hash<4> = 6;

template<>
constexpr int __chars_to_hash<8> = 13;
#endif

template<class HASH>
constexpr int chars_in_type = __chars_to_hash<sizeof(HASH)>;

template<class HASH, class CHAR>
__device__ __host__ __inline__ typename std::enable_if<std::is_integral<HASH>::value &&
														(std::is_same<CHAR, char>::value || std::is_same<CHAR, unsigned char>::value) &&
														(chars_in_type<HASH> > 0), HASH>::type get_hash(
	const CHAR* words, const int my_position)
{
	const int chars_to_hash = chars_in_type<HASH>;

	typename std::make_unsigned<HASH>::type hash = 0;
	unsigned char last_bit = 1;
	unsigned char char_mask = CHARMASK;

	int i = 0;
	for (; i < chars_to_hash; i++)
	{
		const unsigned char c = words[i + my_position];
		if (c == BREAKCHAR)
		{
			char_mask = 0;
			last_bit = 0;
			break;
		}
		hash *= ALPHABETSIZE;
		hash += c & char_mask;
	}
	for (; i < chars_to_hash; i++)
	{
		hash *= ALPHABETSIZE;
	}
	if (!char_mask || words[chars_to_hash + my_position] == BREAKCHAR)
		last_bit = 0;

	return static_cast<HASH>(hash << 1 | last_bit);
}

__device__ __host__ __inline__ ullong get_hash(const uchar* words, const int chars_to_hash, const int my_position)
{
	uchar last_bit = 1;
	uchar char_mask = CHARMASK;

	ullong hash = 0;
	int i = 0;
	for (; i < chars_to_hash; i++)
	{
		const unsigned char c = words[i + my_position];
		if (c == BREAKCHAR)
		{
			char_mask = 0;
			last_bit = 0;
			++i;
			break;
		}
		hash *= ALPHABETSIZE;
		hash += c & char_mask;
	}
	for (; i < chars_to_hash; i++)
	{
		hash *= ALPHABETSIZE;
	}
	if (!char_mask || words[chars_to_hash + my_position] == BREAKCHAR)
		last_bit = 0;

	return hash << 1 | last_bit;
}

inline void compute_grid_size(uint n, uint block_size, uint &num_blocks, uint &num_threads)
{
	num_threads = block_size < n ? block_size : n;
	num_blocks = (n % num_threads != 0) ? (n / num_threads + 1) : (n / num_threads);
}
