#include "dictionary_reader.h"
#include "gpu_test.cuh"
#include "functions.h"

dictionary_reader::dictionary_reader(const char* fileName)
{
	auto dictionaryArray = read_file_to_buffer(fileName);
	std::replace(dictionaryArray.begin(), dictionaryArray.end(), '\n', '\0');
	dictionaryArray.push_back('\0');
	auto it = dictionaryArray.begin();
	auto previous = dictionaryArray.begin();
	it = std::find(it, dictionaryArray.end(), '\0');
	while (it != dictionaryArray.end())
	{
		words.emplace_back(std::string(&*previous));
		previous = it + 1;
		it = std::find(it + 1, dictionaryArray.end(), '\0');
	}
	return;
}

dictionary_reader::dictionary_reader(const std::string& fileName) : dictionary_reader(fileName.c_str())
{
}

dictionary_reader::dictionary_reader(const std::string&& fileName) : dictionary_reader(fileName)
{
}

size_t dictionary_reader::words_count() const
{
	return words.size();
}
