#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iterator>
#include <vector>
#include <ctype.h>
#include <regex>
#include <string>

#define FILEPATH "book.txt"

using namespace std;

bool ReadFile()
{
	unsigned char * buffer = 0;
	long length;
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

	int startingPosition = 0;
	while (true)
	{
		unsigned char c = buffer[startingPosition];
		if (isalpha(c))
			break;
		startingPosition++;
	}

	vector<int> wordPositions;
	wordPositions.push_back(0);
	int writeIndex = 0;
	bool currentlyOnNotAlphaSeq = false;
	for (int i = startingPosition; i < length; i++)
	{
		unsigned char c = buffer[i];

		if ((c > 64 && c < 91) || (c > 96 && c < 123))
		{
			if (currentlyOnNotAlphaSeq)
			{
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
	return true;
}



int main(int argc, char **argv)
{
	findCudaDevice(argc, (const char **)argv);
	ReadFile();
}

