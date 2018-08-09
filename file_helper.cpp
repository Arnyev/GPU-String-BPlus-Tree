#include "parameters.h"
#include <thrust/host_vector.h>
#include <ctime>
#include <iomanip>
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

void create_strings(const thrust::host_vector<uchar>& words_book, const thrust::host_vector<int>& positions_book, std::vector<std::string>& strings_book)
{
	strings_book.resize(positions_book.size());
	std::vector<uchar> chars;

	for (size_t i = 0; i < strings_book.size(); i++)
	{
		const auto position = positions_book[i];
		int index_in_word = 0;
		while (true)
		{
			const auto c = words_book[position + index_in_word];
			if (c != BREAKCHAR)
			{
				chars.push_back(c);
				index_in_word++;
			}
			else
			{
				strings_book[i] = std::string(chars.begin(), chars.end());
				chars.clear();
				break;
			}
		}
	}
}

void append_to_csv(const char* algorithm, const float build_time, const float execution_time,
	const size_t dict_size, const size_t input_size, const double existing_percentage)
{
	int device;
	cudaGetDevice(&device);

	struct cudaDeviceProp props {};
	cudaGetDeviceProperties(&props, device);

	std::ofstream outfile;

	const auto time_point = std::chrono::system_clock::now();

	const auto time = std::chrono::system_clock::to_time_t(time_point);
	struct tm timeinfo{};
	localtime_s(&timeinfo, &time);

	outfile.open("results.csv", std::ios_base::app);
	outfile << std::put_time(&timeinfo, "%c") << ",\t\t" << algorithm << ",\t\t" << props.name << ",\t\t" << build_time << ",\t\t" <<
		execution_time << ",\t\t" << dict_size << ",\t\t" << input_size << ",\t\t" << existing_percentage << std::endl;
}
