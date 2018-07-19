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

bool read_file_to_buffer(size_t &length, unsigned char * &buffer)
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

bool ReadFile(int*& h_word_positions, int*& h_wordLengths, vector<int>& wordPositions, vector<int>& wordLengths, unsigned char *&h_wordArray, int& wordCount, int& charCount)
{
	unsigned char * buffer = 0;
	size_t length;
	if (!read_file_to_buffer(length, buffer))
		return false;

	const int starting_position = GetStartingPosition(buffer);

	wordPositions.push_back(0);
	int writeIndex = 0;
	int wordStart = 0;
	bool currentlyOnNotAlphaSeq = false;
	for (ullong i = starting_position; i < length; i++)
	{
		const unsigned char c = buffer[i];

		if (c > 64 && c < 91 || c > 96 && c < 123)//is alpha
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
			buffer[writeIndex] = BREAKCHAR;
			writeIndex++;
			currentlyOnNotAlphaSeq = true;
		}
	}
	wordLengths.push_back(writeIndex - wordStart);
	buffer[writeIndex] = BREAKCHAR;
	h_word_positions = wordPositions.data();
	h_wordArray = buffer;
	h_wordLengths = wordLengths.data();
	wordCount = wordLengths.size();
	charCount = writeIndex + 1;
	return true;
}

int main(int argc, char **argv)
{
	using hash = ullong;
	std::vector<std::string> words;
	words.emplace_back("abc");
	words.emplace_back("axxxxxxxxxxxxtomek");
	words.emplace_back("axxxxxxxxxxxxjanek");
	words.emplace_back("domek");
	words.emplace_back("romek");
	words.emplace_back("xds");
	std::vector<hash> hashes;
	std::vector<int> indexes;
	std::vector<char> suffixes;
	std::string concat;
	std::vector<int> beginIndexes;
	const char nullbyte = static_cast<char>(0);
	int suffixIndex = 0;
	int nextIndex = 0;
	for (auto& word : words)
	{
		int length = 0;
		hash h = get_hash(reinterpret_cast<const uchar*>(word.c_str()), CHARSTOHASH, 0);
		if (word.length() > CHARSTOHASH)
		{
			for (auto it = word.begin() + CHARSTOHASH; it != word.end(); ++it)
			{
				suffixes.emplace_back(*it);
				++length;
			}
		}
		suffixes.emplace_back(nullbyte);
		++length;
		if (hashes.empty() || h != hashes.back())
		{
			hashes.emplace_back(h);
			indexes.push_back(suffixIndex);
		}
		suffixIndex += length;
		concat.append(word);
		concat.push_back(nullbyte);
		beginIndexes.emplace_back(nextIndex);
		nextIndex = word.size();
	}
	bplus_tree_cpu<hash, 4> tree(hashes.data(), indexes.data(), hashes.size(), suffixes.data(), suffixes.size());
	tree.exist_word("axxxxxxxxxxxxjanex");
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

	cout << "Randoms" << endl;
	test_random_strings();

	cout << "Moby Dick" << endl;

	unsigned char* d_wordArray;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_wordArray), charCount + CHARSTOHASH));
	checkCudaErrors(cudaMemcpy(d_wordArray, h_wordArray, charCount, cudaMemcpyHostToDevice));

	int* d_wordPositions;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_wordPositions), sizeof(int)*wordCount));
	checkCudaErrors(cudaMemcpy(d_wordPositions, h_wordPositions, sizeof(int)*wordCount, cudaMemcpyHostToDevice));

	int* d_sorted_positions = get_sorted_positions(d_wordPositions, wordCount, d_wordArray);
	const bool sorting_result = test_string_sorting(d_sorted_positions, wordCount, h_wordArray, charCount);

	const auto output = create_output(d_wordArray, d_sorted_positions, wordCount);
	const bool output_result = test_output(h_wordArray, charCount, output);

	if (!output_result || !sorting_result)
	{
		cout << "Fail" << endl;
		throw exception();
	}
	else
		cout << "Win" << endl;
}
