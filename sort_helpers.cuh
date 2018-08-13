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
__device__ __host__ __inline__ typename std::enable_if<
	std::is_integral<HASH>::value && (std::is_same<CHAR, char>::value || std::is_same<CHAR, unsigned char>::value) && (
		chars_in_type<HASH> > 0), HASH>::type get_hash(const CHAR* words, const int my_position)
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

template<class HASH, class CHAR>
__device__ __host__ __inline__ typename std::enable_if<
	std::is_integral<HASH>::value && (std::is_same<CHAR, char>::value || std::is_same<CHAR, unsigned char>::value) && (
		chars_in_type<HASH> > 0), HASH>::type get_hash_v2(const CHAR* words, const int my_position)
{
	constexpr int chars_to_hash = chars_in_type<HASH>;
	using UnHash = typename std::make_unsigned<HASH>::type;
	constexpr int sizeOfLookup = sizeof(uint32_t);
	UnHash hash = 0;
	constexpr uint32_t idx1word = 0x00'00'00'1Fu;
	constexpr uint32_t idx2word = 0x00'00'1F'00u;
	constexpr uint32_t idx3word = 0x00'1F'00'00u;
	constexpr uint32_t idx4word = 0x1F'00'00'00u;
	const uint32_t *wordPtr = reinterpret_cast<const uint32_t*>(words + my_position);
	int i = 0;
	while (i < chars_to_hash)
	{
		const uint32_t pack4chars = *wordPtr;
		if (pack4chars & idx1word)
		{
			hash *= ALPHABETSIZE;
			hash += pack4chars & idx1word;
			++i;
		}
		else
			break;
		if (pack4chars & idx2word && i < chars_to_hash)
		{
			hash *= ALPHABETSIZE;
			hash += (pack4chars & idx2word) >> 8;
			++i;
		}
		else
			break;
		if (pack4chars & idx3word && i < chars_to_hash)
		{
			hash *= ALPHABETSIZE;
			hash += (pack4chars & idx3word) >> 16;
			++i;
		}
		else
			break;
		if (pack4chars & idx4word && i < chars_to_hash)
		{
			hash *= ALPHABETSIZE;
			hash += (pack4chars & idx4word) >> 24;
			++i;
		}
		else
			break;
		++wordPtr;
	}
	bool wordExceeds = false;
	if (i == chars_to_hash)
	{
		switch (chars_to_hash % sizeOfLookup)
		{
		case 0:
			wordExceeds = *wordPtr & idx1word;
			break;
		case 1:
			wordExceeds = *wordPtr & idx2word;
			break;
		case 2:
			wordExceeds = *wordPtr & idx3word;
			break;
		case 3:
			wordExceeds = *wordPtr & idx4word;
			break;
		default: ;
		}
	}
	else
	{
		while (i < chars_to_hash)
		{
			hash *= ALPHABETSIZE;
			++i;
		}
	}
	hash <<= 1;
	hash |= wordExceeds ? 0x1 : 0x0;
	return static_cast<HASH>(hash);
}

template<class HASH, class CHAR>
__device__ __host__ __inline__ typename std::enable_if<
	std::is_integral<HASH>::value && (std::is_same<CHAR, char>::value || std::is_same<CHAR, unsigned char>::value) && (
		chars_in_type<HASH> > 0), HASH>::type get_hash_v3(const CHAR* words, const int my_position)
{
	using UnHash = typename std::make_unsigned<HASH>::type;
	UnHash hash = 0;
	constexpr uint32_t idx1word = 0x00'00'00'1Fu;
	constexpr uint32_t idx2word = 0x00'00'1F'00u;
	constexpr uint32_t idx3word = 0x00'1F'00'00u;
	constexpr uint32_t idx4word = 0x1F'00'00'00u;
	const uint4 pack16chars = *reinterpret_cast<const uint4*>(words + my_position);
	bool end = false;
	do
	{
#define checkNthChar(N, a)\
			end = end || !(idx##N##word & pack16chars.##a);\
			hash *= ALPHABETSIZE;\
			hash += end ? 0 : ((idx##N##word & pack16chars.##a) >> (8 * (N - 1)));

		checkNthChar(1, x);
		checkNthChar(2, x);
		checkNthChar(3, x);
		checkNthChar(4, x);
		checkNthChar(1, y);
		checkNthChar(2, y);
		checkNthChar(3, y);
		checkNthChar(4, y);
		checkNthChar(1, z);
		checkNthChar(2, z);
		checkNthChar(3, z);
		checkNthChar(4, z);
		checkNthChar(1, w);
#undef checkNthChar
	} while (false);
	end = end || !(idx2word & pack16chars.w);
	hash <<= 1;
	hash |= end ? 0x0 : 0x1;
	return static_cast<HASH>(hash);
}
