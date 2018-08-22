#include "bplus_tree_cpu.h"
#include "functions.h"
#include <helper_cuda.h>
#include "gpu_test.cuh"
#include "dictionary_reader.h"
#include "book_reader.h"

using namespace std;

void get_arguments(const int argc, char **argv, std::string &dictionary, std::string &book, int &iterations);

int main(const int argc, char **argv)
{
	findCudaDevice(argc, const_cast<const char **>(argv));

	int* test;//initialization to improve time testing accuracy
	if (cudaMalloc(&test, 4 * 4))
		return 0;

	std::string book_file = "terminalCompromise.txt";
	std::string dictionary_file = "dictionary_clean.txt";
	int iterations = 1;
	get_arguments(argc, argv, dictionary_file, book_file, iterations);
	dictionary_reader dict(dictionary_file);
	book_reader book(book_file);
	for (int i = 0; i < iterations; ++i)
	{
		//test_gpu_tree<uint64_t, 4, 1>(dict, book);
		//test_gpu_tree<uint64_t, 4096, 2>(dict, book);
		//test_gpu_tree<uint64_t, 8192, 2>(dict, book);
		test_gpu_tree<uint64_t, 1024, 3>(dict, book);
		test_gpu_tree<uint64_t, 2048, 3>(dict, book);
		//test_gpu_tree<uint64_t, 4096, 3>(dict, book);
		//test_gpu_tree<uint64_t, 8192, 3>(dict, book);
		test_gpu_tree<uint64_t, 1024, 4>(dict, book);
		test_gpu_tree<uint64_t, 2048, 4>(dict, book);
		//test_gpu_tree<uint64_t, 4096, 4>(dict, book);
		//test_gpu_tree<uint64_t, 8192, 4>(dict, book);
		test_gpu_tree<uint64_t, 16, 5>(dict, book);
		test_gpu_tree<uint64_t, 32, 5>(dict, book);
		test_gpu_tree<uint64_t, 64, 5>(dict, book);
		test_gpu_tree<uint64_t, 128, 5>(dict, book);
		test_gpu_tree<uint64_t, 256, 5>(dict, book);
		//test_gpu_tree<uint64_t, 512, 5>(dict, book);
		//test_gpu_tree<uint64_t, 1024, 5>(dict, book);
		//test_gpu_tree<uint64_t, 2048, 5>(dict, book);
		//test_gpu_tree<uint64_t, 4096, 5>(dict, book);
		//test_gpu_tree<uint64_t, 8192, 5>(dict, book);
		test_gpu_tree<uint64_t, 16, 6>(dict, book);
		test_gpu_tree<uint64_t, 32, 6>(dict, book);
		test_gpu_tree<uint64_t, 64, 6>(dict, book);
		test_gpu_tree<uint64_t, 128, 6>(dict, book);
		test_gpu_tree<uint64_t, 256, 6>(dict, book);
		//test_gpu_tree<uint64_t, 512, 6>(dict, book);
		//test_gpu_tree<uint64_t, 1024, 6>(dict, book);
		//test_gpu_tree<uint64_t, 2048, 6>(dict, book);
		//test_gpu_tree<uint64_t, 4096, 6>(dict, book);
		//test_gpu_tree<uint64_t, 8192, 6>(dict, book);
		test_array_searching_book("dictionary_clean.txt", "terminalCompromise.txt");
	}
	return 0;
	test_array_searching_book("dictionary.txt", "book.txt");

	cout << "Randoms" << endl;
	test_random_strings();

	cout << "Moby Dick" << endl;
	test_book("book.txt");
}

void get_arguments(const int argc, char **argv, std::string &dictionary, std::string &book, int &iterations)
{
	constexpr char delimiter = '-';
	const char dictionaryFlag[] = "dict";
	const char bookFlag[] = "book";
	const char iterationsFlag[] = "n";
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
			char *word = std::strstr(pos, dictionaryFlag);
			if (word != nullptr && *(word += std::extent<decltype(dictionaryFlag)>::value - 1) == '=')
			{
				++word;
				dictionary.assign(word);
				continue;
			}
			word = std::strstr(pos, bookFlag);
			if (word != nullptr && *(word += std::extent<decltype(bookFlag)>::value - 1) == '=')
			{
				++word;
				book.assign(word);
				continue;
			}
			word = std::strstr(pos, iterationsFlag);
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

