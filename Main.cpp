#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>
#include <cctype>
#include <regex>
#include "Main.h"
#include "bplus_tree_gpu.cuh"
#include "bplus_tree_cpu.h"
#include "parameters.h"
#include "functions.h"

using namespace std;

bool  ReadFileToBuffer(size_t &length, unsigned char * &buffer)
{
	FILE * f = fopen(FILEPATH, "rb");

	if (!f)
		return false;

	fseek(f, 0, SEEK_END);
	length = ftell(f);
	fseek(f, 0, SEEK_SET);
	buffer = static_cast<unsigned char*>(malloc(length));
	if (!buffer)
		return false;

	if (fread(buffer, 1, length, f) != length)
		return false;

	fclose(f);
	return true;
}

int GetStartingPosition(const unsigned char * buffer)
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
	size_t length;
	if (!ReadFileToBuffer(length, buffer))
		return false;

	const int starting_position = GetStartingPosition(buffer);

	wordPositions.push_back(0);
	int writeIndex = 0;
	int wordStart = 0;
	bool currentlyOnNotAlphaSeq = false;
	for (ullong i = starting_position; i < length; i++)
	{
		unsigned char c = buffer[i];

		if ((c > 64 && c < 91) || (c > 96 && c < 123))//is alpha
		{
			if (currentlyOnNotAlphaSeq)
			{
				wordLengths.push_back(writeIndex - wordStart - 1);
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
	wordLengths.push_back(writeIndex - wordStart);
	buffer[writeIndex] = ' ';
	h_wordPositions = wordPositions.data();
	h_wordArray = buffer;
	h_wordLengths = wordLengths.data();
	wordCount = wordLengths.size();
	charCount = writeIndex + 1;
	return true;
}

void test_random_strings()
{
	vector<vector<char>> strings(1000000);
	int len = 0;
	const char charset[] = "abc";
	const int sizer = sizeof charset - 1;
	for (int j = 0; j < 1000000; j++)
	{
		const int lena = rand() % RANDSTRMAXLEN + 1;
		strings[j].reserve(lena);
		for (int i = 0; i < lena; i++)
			strings[j].push_back(charset[rand() % sizer]);

		len += strings[j].size() + 1;
	}
	vector<char> vecc{};
	vecc.reserve(len);
	vector<int> positions(strings.size());
	vector<int> lengths(strings.size());
	vector<char> chars(len);
	int currentPosition = 0;
	for (uint k = 0; k < strings.size(); k++)
	{
		positions[k] = currentPosition;
		lengths[k] = strings[k].size();
		for (int l = 0; l < lengths[k]; l++)
			chars[currentPosition++] = strings[k][l];
		chars[currentPosition++] = ' ';
	}
	int* d_positions = get_sorted_positions(reinterpret_cast<unsigned char*>(chars.data()), positions.data(), lengths.data(), lengths.size(), chars.size());

	const bool sorting_result = test_string_sorting(d_positions, lengths.size(), reinterpret_cast<unsigned char*>(chars.data()), chars.size());

	unsigned char* d_wordArray;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_wordArray), chars.size()));
	checkCudaErrors(cudaMemcpy(d_wordArray, chars.data(), chars.size(), cudaMemcpyHostToDevice));

	const auto output = create_output(d_wordArray, d_positions, lengths.size());

	const bool output_result = test_output(reinterpret_cast<uchar*>(chars.data()), chars.size(), output);

	if (!output_result || !sorting_result)
		throw exception();
}

int main(int argc, char **argv)
{
	//Sample creation of B+ tree
	//int i = 0;
	//int size = 64;
	//vector<int> keys(size);
	//vector<int> values(size);
	//generate(keys.begin(), keys.end(), [&i]() -> int { return ++i; });
	//i = 1;
	//generate(values.begin(), values.end(), [&i]() -> int { return ++i; });
	//bplus_tree_gpu<int, 4> gt(keys.data(), values.data(), size);
	//bplus_tree_cpu<int, 4> ct(gt);
	//bplus_tree_cpu<int, 4> cte(keys.data(), values.data(), size);
	//int toFind[] = { 1, 2, 0, -1, 64, 65, 3 };
	//auto z1 = ct.get_value(toFind, sizeof(toFind) / sizeof(int));
	//auto z2 = cte.get_value(toFind, sizeof(toFind) / sizeof(int));
	//auto z3 = gt.get_value(toFind, sizeof(toFind) / sizeof(int));
	//ct and cte should be equal

	vector<int> word_positions;
	vector<int> word_lengths;

	findCudaDevice(argc, const_cast<const char **>(argv));
	int* h_wordPositions;
	int* h_wordLengths;
	unsigned char *h_wordArray;
	int wordCount;
	int charCount;
	ReadFile(h_wordPositions, h_wordLengths, word_positions, word_lengths, h_wordArray, wordCount, charCount);

	test_random_strings();

	int* d_positions = get_sorted_positions(h_wordArray, h_wordPositions, h_wordLengths, wordCount, charCount);

	const bool sorting_result = test_string_sorting(d_positions, wordCount, h_wordArray, charCount);

	unsigned char* d_wordArray;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_wordArray), charCount));
	checkCudaErrors(cudaMemcpy(d_wordArray, h_wordArray, charCount, cudaMemcpyHostToDevice));

	const auto output = create_output(d_wordArray, d_positions, wordCount);

	const bool output_result = test_output(h_wordArray, charCount, output);

	if (!output_result || !sorting_result)
		throw exception();
}
