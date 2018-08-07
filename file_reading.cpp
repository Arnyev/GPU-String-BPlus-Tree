#include "parameters.h"
#include <locale>
#include <thrust/host_vector.h>
#include <fstream> 
#include <cctype>

std::vector<char> read_file_to_buffer(const char* filepath)
{
	std::ifstream file(filepath, std::ios::binary | std::ios::ate);
	const std::streamsize size = file.tellg();
	if (size <= 0)
	{
		std::cout << "Fail reading file " << filepath;
		exit(EXIT_FAILURE);
	}

	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size))
	{
		std::cout << "Fail reading file " << filepath;
		exit(EXIT_FAILURE);
	}

	return buffer;
}

void read_file(const char* filepath, thrust::host_vector<int>& positions, thrust::host_vector<uchar>& words)
{
	auto chars = read_file_to_buffer(filepath);

	auto c = '\0';

	size_t starting_position = 0;
	for (; !isalpha(c) && starting_position < chars.size(); starting_position++)
		c = chars[starting_position];

	starting_position--;
	positions.push_back(0);
	auto currently_on_not_alpha_seq = false;

	const uchar mask = TOLOWERMASK;
	for (auto i = starting_position; i < chars.size(); i++)
	{
		c = chars[i];

		if (c > 0 && std::isalpha(c))
		{
			if (currently_on_not_alpha_seq)
			{
				positions.push_back(static_cast<int>(words.size()));
				currently_on_not_alpha_seq = false;
			}

			words.push_back(c | mask);
		}
		else if (!currently_on_not_alpha_seq)
		{
			words.push_back(BREAKCHAR);
			currently_on_not_alpha_seq = true;
		}
	}

	for (int i = 0; i < CHARSTOHASH; i++)
		words.push_back(BREAKCHAR);
}