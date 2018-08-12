#include "functions.h"
#include "bplus_tree_gpu.cuh"
#include <sstream>

template <typename HASH, int PAGE_SIZE, int Version>
void create_tree(thrust::device_vector<int>& positions_dictionary, const thrust::device_vector<char>& words_dictionary,
	bplus_tree_gpu<HASH, PAGE_SIZE>& tree)
{
	tree.create_tree(words_dictionary, positions_dictionary);
}

template <typename HASH, int PAGE_SIZE, int Version>
void exist_word_tree(const thrust::device_vector<int>& positions_book, const thrust::device_vector<char>& words_book,
	const bplus_tree_gpu<HASH, PAGE_SIZE>& tree, thrust::device_vector<bool>& gpu_result)
{
	tree.template exist_word<Version>(words_book, positions_book, gpu_result);
}

template<typename HASH, int PAGE_SIZE, int Version>
void test_gpu_tree(const char* dictionary_filename, const char* book_filename, bool showMissingWords = false)
{
	thrust::host_vector<int> positions_dictionary_host;
	thrust::host_vector<char> words_dictionary_host;
	read_file(dictionary_filename, positions_dictionary_host, words_dictionary_host);

	thrust::device_vector<char> words_dictionary = words_dictionary_host;

	thrust::device_vector<int> positions_dictionary;
	positions_dictionary.reserve(positions_dictionary_host.size() + 1);
	positions_dictionary.resize(positions_dictionary_host.size());
	thrust::copy(positions_dictionary_host.begin(), positions_dictionary_host.end(), positions_dictionary.begin());

	bplus_tree_gpu<HASH, PAGE_SIZE> tree;

	const auto build_time = measure::execution_gpu(create_tree<HASH, PAGE_SIZE, Version>, positions_dictionary, words_dictionary, tree);

	thrust::host_vector<int> positions_book_host;
	thrust::host_vector<char> words_book_host;
	read_file(book_filename, positions_book_host, words_book_host);

	const thrust::device_vector<int> positions_book = positions_book_host;
	const thrust::device_vector<char> words_book = words_book_host;

	thrust::device_vector<bool> gpu_result;
	const auto execution_time = measure::execution_gpu(exist_word_tree<HASH, PAGE_SIZE, Version>, positions_book, words_book, tree, gpu_result);

	std::vector<std::string> strings;
	get_strings(words_book_host, positions_book_host, strings);

	auto result = from_vector_dev(gpu_result);

	std::vector<bool> cpu_result;
	search_cpu(words_dictionary_host, positions_dictionary_host, words_book_host, positions_book_host, cpu_result);

	int true_count = 0;
	for (size_t i = 0; i < result.size(); i++)
	{
		if (cpu_result[i] != result[i] && showMissingWords)
			std::cout << strings[i] << std::endl;

		if (result[i])
			true_count++;
	}

	std::stringstream stream;
	stream << "bplus_tree_gpu<" << typeid(HASH).name() << ", " << PAGE_SIZE << "> v" << Version;

	append_to_csv(stream.str().c_str(), build_time / 1000, execution_time / 1000, positions_dictionary.size(),
		result.size(), static_cast<double>(true_count) / result.size());
}

void test_tree()
{
	test_gpu_tree<uint64_t, 4, 1>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 8, 1>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 16, 1>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 32, 1>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 512, 2>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 1024, 2>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 2048, 2>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 4096, 2>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 8192, 2>("dictionary_clean.txt", "oliverTwist.txt");
	test_gpu_tree<uint64_t, 16384, 2>("dictionary_clean.txt", "oliverTwist.txt");
}
