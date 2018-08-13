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
	std::vector<int> s;
	dictionary_reader dict("dictionary_clean.txt");
	book_reader book("olivertwist.txt");
	test_gpu_tree<uint64_t, 4, 1>(dict, book);
	test_gpu_tree<uint64_t, 4096, 2>(dict, book);
	test_gpu_tree<uint64_t, 8192, 2>(dict, book);
	test_gpu_tree<uint64_t, 16384, 2>(dict, book);
	test_gpu_tree<uint64_t, 1024, 3>(dict, book);
	test_gpu_tree<uint64_t, 2048, 3>(dict, book);
	test_gpu_tree<uint64_t, 4096, 3>(dict, book);
	test_gpu_tree<uint64_t, 8192, 3>(dict, book);
	test_gpu_tree<uint64_t, 16384, 3>(dict, book);
	test_gpu_tree<uint64_t, 1024, 4>(dict, book);
	test_gpu_tree<uint64_t, 2048, 4>(dict, book);
	test_gpu_tree<uint64_t, 4096, 4>(dict, book);
	test_gpu_tree<uint64_t, 8192, 4>(dict, book);
	test_gpu_tree<uint64_t, 16384, 4>(dict, book);
	test_array_searching_book("dictionary_clean.txt", "oliverTwist.txt");
	return 0;
	test_array_searching_book("dictionary.txt", "book.txt");

	cout << "Randoms" << endl;
	test_random_strings();

	cout << "Moby Dick" << endl;
	test_book("book.txt");
}
