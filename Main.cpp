#include "bplus_tree_cpu.h"
#include "functions.h"
#include <helper_cuda.h>
#include "gpu_test.cuh"
#include "dictionary_reader.h"
#include "book_reader.h"

using namespace std;

void get_arguments(const int argc, char **argv, std::string &dictionary, std::string &book, int &iterations, std::string &csv);

int main(const int argc, char **argv)
{
	findCudaDevice(argc, const_cast<const char **>(argv));

	int* test;//initialization to improve time testing accuracy
	if (cudaMalloc(&test, 4 * 4))
		return 0;

	std::string book_file = "terminalCompromise.txt";
	std::string dictionary_file = "dictionary_clean.txt";
	std::string csv_file = "results.csv";
	int iterations = 1;
	get_arguments(argc, argv, dictionary_file, book_file, iterations, csv_file);
	{
		dictionary_reader dict(dictionary_file);
		book_reader book(book_file);
		csv_logger logger(csv_file);
		for (int i = 0; i < iterations; ++i)
		{
			test_array_searching_book(dict, book, logger);
			test_gpu_tree<uint64_t, 1024, 3>(dict, book, logger);
			test_gpu_tree<uint64_t, 2048, 3>(dict, book, logger);
			test_gpu_tree<uint64_t, 1024, 4>(dict, book, logger);
			test_gpu_tree<uint64_t, 2048, 4>(dict, book, logger);
			test_gpu_tree<uint64_t, 16, 5>(dict, book, logger);
			test_gpu_tree<uint64_t, 32, 5>(dict, book, logger);
			test_gpu_tree<uint64_t, 64, 5>(dict, book, logger);
			test_gpu_tree<uint64_t, 128, 5>(dict, book, logger);
			test_gpu_tree<uint64_t, 256, 5>(dict, book, logger);
			test_gpu_tree<uint64_t, 16, 6>(dict, book, logger);
			test_gpu_tree<uint64_t, 32, 6>(dict, book, logger);
			test_gpu_tree<uint64_t, 64, 6>(dict, book, logger);
			test_gpu_tree<uint64_t, 128, 6>(dict, book, logger);
			test_gpu_tree<uint64_t, 256, 6>(dict, book, logger);
			test_gpu_tree<uint64_t, 16, 7>(dict, book, logger);
			test_gpu_tree<uint64_t, 32, 7>(dict, book, logger);
			test_gpu_tree<uint64_t, 64, 7>(dict, book, logger);
			test_gpu_tree<uint64_t, 128, 7>(dict, book, logger);
			test_gpu_tree<uint64_t, 256, 7>(dict, book, logger);
			std::cout << "Done " << i + 1 << " out of " << iterations << " iterations.\n";
		}
	}
	return 0;
}


template <typename ConstCharArray>
inline bool assign_if_equal(char* pos, std::string &str, ConstCharArray &flag)
{
	char *word = std::strstr(pos, flag);
	if (word != nullptr && *(word += std::extent<ConstCharArray>::value - 1) == '=')
	{
		++word;
		str.assign(word);
		return true;
	}
	return false;
}

void get_arguments(const int argc, char **argv, std::string &dictionary, std::string &book, int &iterations, std::string &csv)
{
	constexpr char delimiter = '-';
	const char dictionaryFlag[] = "dict";
	const char bookFlag[] = "book";
	const char iterationsFlag[] = "n";
	const char csvFlag[] = "csv";
	for (int i = 1; i < argc; ++i)
	{
		char *it = argv[i];
		char *pos = nullptr;
		while (*it)
		{
			if (*it == delimiter)
				pos = it;
			++it;
		}
		if (pos != nullptr)
		{
			++pos;
			if (assign_if_equal(pos, dictionary, dictionaryFlag))
				continue;
			if (assign_if_equal(pos, book, bookFlag))
				continue;
			if (assign_if_equal(pos, csv, csvFlag))
				continue;;
			char* word = std::strstr(pos, iterationsFlag);
			if (word != nullptr && *(word += std::extent<decltype(iterationsFlag)>::value - 1) == '=')
			{
				++word;
				const int result = std::atoi(word);
				if (result != 0)
					iterations = result;
			}
		}
	}
}

