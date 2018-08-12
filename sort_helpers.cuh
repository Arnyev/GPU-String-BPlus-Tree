#pragma once

#include <parameters.h>
#include <type_traits>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <helper_cuda.h>

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

//get_hash expects the char array to have at least chars_to_hash elements after my_position index
template<class HASH, class CHAR>
__device__ __host__ __inline__ typename std::enable_if<
	std::is_integral<HASH>::value && (std::is_same<CHAR, char>::value || std::is_same<CHAR, unsigned char>::value) && (
		chars_in_type<HASH> > 0), HASH>::type get_hash(const CHAR* words, const int my_position, const int chars_to_hash)
{
	typename std::make_unsigned<HASH>::type hash = 0;
	CHAR last_bit = 1;
	CHAR char_mask = CHARMASK;

	for (int i = 0; i < chars_to_hash; i++)
	{
		const CHAR c = words[i + my_position];
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

	return static_cast<HASH>(hash << 1 | last_bit);
}

template<class HASH, class CHAR>
__device__ __host__ __inline__ typename std::enable_if<
	std::is_integral<HASH>::value && (std::is_same<CHAR, char>::value || std::is_same<CHAR, unsigned char>::value) && (
		chars_in_type<HASH> > 0), HASH>::type get_hash(const CHAR* words, const int my_position)
{
	const int chars_to_hash = chars_in_type<HASH>;

	typename std::make_unsigned<HASH>::type hash = 0;
	CHAR last_bit = 1;
	CHAR char_mask = CHARMASK;

	for (int i = 0; i < chars_to_hash; i++)
	{
		const CHAR c = words[i + my_position];
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

	return static_cast<HASH>(hash << 1 | last_bit);
}

template<class HASH> typename std::enable_if<std::is_integral<HASH>::value, HASH>::type get_hash_str(const std::string& word)
{
	int i = 0;
	HASH hash = 0;
	for (; i < CHARSTOHASH; i++)
	{
		const unsigned char c = word[i];
		if (c == '\0')
			break;

		hash *= ALPHABETSIZE;
		hash += c & CHARMASK;
	}

	const HASH mask = word[i] == '\0' ? 0 : 1;

	for (; i < CHARSTOHASH; i++)
		hash *= ALPHABETSIZE;

	return static_cast<HASH>(hash << 1 | mask);
}

template<class T, T N>
struct equal_to_val : thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T x) const { return x == N; }
};

struct less_than_string : thrust::binary_function<int, int, bool>
{
	const char * words;
	explicit less_than_string(const char* words) : words(words) {	}

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

template<class HASH>
struct hash_functor : thrust::unary_function<int, HASH>
{
	const char* words;

	explicit hash_functor(const char* words) : words(words) {	}

	__host__ __device__ HASH operator()(const int position) const
	{
		return get_hash<HASH>(words, position);
	}
};

struct compute_postfix_length_functor : thrust::unary_function<int, int>
{
	const char* words;

	explicit compute_postfix_length_functor(const char* words) : words(words) {}

	__device__  int operator()(int my_position) const
	{
		if (my_position == -1)
			return 0;

		int length = 0;
		char c;
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
