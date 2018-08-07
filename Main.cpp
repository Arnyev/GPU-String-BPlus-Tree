#include "bplus_tree_cpu.h"
#include "functions.h"
#include "sort_strings.cuh"
#include <helper_cuda.h>

using namespace std;

int main(const int argc, char **argv)
{
	findCudaDevice(argc, const_cast<const char **>(argv));

	int* test;//initialization to improve time testing accuracy
	if (cudaMalloc(&test, 4 * 4))
		return 0;

	test_gpu_tree("dictionary_clean.txt", "oliverTwist.txt");
	test_array_searching_book("dictionary.txt", "oliverTwist.txt");

	cout << "Randoms" << endl;
	test_random_strings();

	cout << "Moby Dick" << endl;
	test_book("book.txt");
}
