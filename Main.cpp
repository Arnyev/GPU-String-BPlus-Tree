#include "bplus_tree_cpu.h"
#include "functions.h"
#include <helper_cuda.h>
#include "gpu_test.cuh"
#include "dictionary_reader.h"
#include "book_reader.h"

using namespace std;

int main(const int argc, char **argv)
{
	findCudaDevice(argc, const_cast<const char **>(argv));

	int* test;//initialization to improve time testing accuracy
	if (cudaMalloc(&test, 4 * 4))
		return 0;
	int iterations = 1;
	if (argc > 1)
	{
		try
		{
			iterations = std::stoi(argv[1]);
		}
		catch(...)
		{
		}
	}
	dictionary_reader dict("dictionary_clean.txt");
	book_reader book("terminalCompromise.txt");
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
