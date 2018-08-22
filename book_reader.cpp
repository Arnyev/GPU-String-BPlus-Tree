#include "book_reader.h"
#include "gpu_test.cuh"

book_reader::book_reader(const char* fileName)
{
	auto bookArray = read_file_to_buffer(fileName);
	std::transform(bookArray.begin(), bookArray.end(), bookArray.begin(), [](char c) -> char {
		if (!std::isalpha(c))
			return ' ';
		return std::tolower(c);
	});
	auto itWord = std::find_if(bookArray.begin(), bookArray.end(), ::isalpha);
	auto itSpace = std::find(itWord, bookArray.end(), ' ');
	while (itWord != bookArray.end())
	{
		words.emplace_back(std::string(&*itWord, itSpace - itWord));
		itWord = std::find_if(itSpace, bookArray.end(), ::isalpha);
		itSpace = std::find(itWord, bookArray.end(), ' ');
	}
	return;
}

book_reader::book_reader(const std::string& fileName) : book_reader(fileName.c_str())
{
}

book_reader::book_reader(const std::string&& fileName) : book_reader(fileName)
{
}

std::tuple<const std::vector<char>&, const std::vector<int>&> book_reader::get_words(int align)
{
	assert(align >= 1);
	auto it = wordsOutput.find(align);
	if (it != wordsOutput.end())
	{
		return it->second;
	}
	auto inserted = wordsOutput.emplace(align, std::make_tuple(std::vector<char>(), std::vector<int>()));
	std::vector<char> &result = std::get<0>(inserted.first->second);
	std::vector<int> &positions = std::get<1>(inserted.first->second);
	for (auto & word : words)
	{
		positions.emplace_back(result.size());
		result.insert(result.end(), word.begin(), word.end());
		const int zeros = align - (word.size() % align);
		result.insert(result.end(), zeros, '\0');
	}
	return {result, positions};
}
