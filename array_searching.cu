#include <thrust/binary_search.h>
#include "sort_helpers.cuh"

using namespace thrust;
void find_if_strings_exist(const device_vector<int>& values_positions, const device_vector<int>& input_positions,
	const device_vector<char>& words, device_vector<bool>& result)
{
	result.resize(values_positions.size());
	const less_than_string func(words.data().get());

	binary_search(input_positions.begin(), input_positions.end(), values_positions.begin(), values_positions.end(), result.begin(), func);
}

void prepare_for_search(const host_vector<int>& positions_dictionary_host, const host_vector<char>& words_dictionary_host,
	const host_vector<int>& positions_book_host, const host_vector<char>& words_book_host, device_vector<int>& positions_book,
	device_vector<char>& words, device_vector<int>& positions_dictionary)
{
	positions_dictionary.resize(positions_dictionary_host.size());
	thrust::copy(positions_dictionary_host.begin(), positions_dictionary_host.end(), positions_dictionary.begin());

	positions_book.resize(positions_book_host.size());
	thrust::copy(positions_book_host.begin(), positions_book_host.end(), positions_book.begin());

	using namespace thrust::placeholders;
	transform(positions_book.begin(), positions_book.end(), positions_book.begin(), _1 + words_dictionary_host.size());

	words.resize(words_dictionary_host.size() + words_book_host.size() + CHARSTOHASH);
	copy(words_dictionary_host.begin(), words_dictionary_host.end(), words.begin());
	copy(words_book_host.begin(), words_book_host.end(), words.begin() + words_dictionary_host.size());
}
