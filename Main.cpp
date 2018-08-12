#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "functions.h"

using namespace std;

int main(const int argc, char **argv)
{
	findCudaDevice(argc, const_cast<const char **>(argv));

	int* test;//initialization to improve time testing accuracy
	if (cudaMalloc(&test, 4 * 4))
		return 0;

	test_tree();
	test_array_searching_book("dictionary_clean.txt", "oliverTwist.txt");
	test_random_strings();
	test_book("book.txt");
}
