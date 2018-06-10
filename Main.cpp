#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iterator>
#include <vector>
#include <ctype.h>
#include <regex>
#include <string>
#include "Main.h"
#include "bplus_tree_gpu.cuh"
#include "bplus_tree_cpu.h"

#define FILEPATH "book.txt"

using namespace std;
void SortStrings(unsigned char* h_wordArray, int* h_wordPositions, int* h_wordLengths, int wordCount, size_t wordArraySize);

bool  ReadFileToBuffer(long &length, unsigned char * &buffer)
{
	FILE * f = fopen(FILEPATH, "rb");

	if (!f)
		return false;

	fseek(f, 0, SEEK_END);
	length = ftell(f);
	fseek(f, 0, SEEK_SET);
	buffer = (unsigned char*)malloc(length);
	if (!buffer)
		return false;

	if (fread(buffer, 1, length, f) != length)
		return false;

	fclose(f);
	return true;
}

int GetStartingPosition(unsigned char * buffer)
{
	int startingPosition = 0;
	while (true)
	{
		unsigned char c = buffer[startingPosition];
		if (isalpha(c))
			break;
		startingPosition++;
	}

	return startingPosition;
}

bool ReadFile(int*& h_wordPositions, int*& h_wordLengths, vector<int>& wordPositions, vector<int>& wordLengths, unsigned char *&h_wordArray, int& wordCount, int& charCount)
{
	unsigned char * buffer = 0;
	long length;
	if (!ReadFileToBuffer(length, buffer))
		return false;

	int startingPosition = GetStartingPosition(buffer);

	wordPositions.push_back(0);
	int writeIndex = 0;
	int wordStart = 0;
	bool currentlyOnNotAlphaSeq = false;
	for (int i = startingPosition; i < length; i++)
	{
		unsigned char c = buffer[i];

		if ((c > 64 && c < 91) || (c > 96 && c < 123))//is alpha
		{
			if (currentlyOnNotAlphaSeq)
			{
				wordLengths.push_back(writeIndex - wordStart);
				wordStart = writeIndex;
				wordPositions.push_back(writeIndex);
				currentlyOnNotAlphaSeq = false;
			}
			buffer[writeIndex] = c;
			writeIndex++;
		}
		else if (!currentlyOnNotAlphaSeq)
		{
			buffer[writeIndex] = ' ';
			writeIndex++;
			currentlyOnNotAlphaSeq = true;
		}
	}
	wordLengths.push_back(writeIndex + 1 - wordStart);
	buffer[writeIndex] = ' ';
	h_wordPositions = wordPositions.data();
	h_wordArray = buffer;
	h_wordLengths = wordLengths.data();
	wordCount = wordLengths.size();
	charCount = writeIndex + 1;
	return true;
}

int main(int argc, char **argv)
{
	//Sample creation of B+ tree
	int i = 0;
	int size = 64;
	vector<int> keys(size);
	vector<int> values(size);
	generate(keys.begin(), keys.end(), [&i]() -> int { return ++i; });
	i = 1;
	generate(values.begin(), values.end(), [&i]() -> int { return ++i; });
	bplus_tree_gpu<int, 4> gt(keys.data(), values.data(), size);
	bplus_tree_cpu<int, 4> ct(gt);
	bplus_tree_cpu<int, 4> cte(keys.data(), values.data(), size);
	int toFind[] = { 1, 2, 0, -1, 64, 65, 3 };
	auto z1 = ct.get_value(toFind, sizeof(toFind) / sizeof(int));
	auto z2 = cte.get_value(toFind, sizeof(toFind) / sizeof(int));
	auto z3 = gt.get_value(toFind, sizeof(toFind) / sizeof(int));
	//ct and cte should be equal

	vector<int> wordPositions;
	vector<int> wordLengths;

	findCudaDevice(argc, (const char **)argv);
	int* h_wordPositions;
	int* h_wordLengths;
	unsigned char *h_wordArray;
	int wordCount;
	int charCount;
	ReadFile(h_wordPositions, h_wordLengths, wordPositions, wordLengths, h_wordArray, wordCount, charCount);

	SortStrings(h_wordArray, h_wordPositions, h_wordLengths, wordCount, charCount);
}

