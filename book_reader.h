#pragma once
#include <vector>
#include <unordered_map>
#include <string>

class book_reader
{
	std::vector<std::string> words;
	std::unordered_map<int, std::tuple<std::vector<char>, std::vector<int>>> wordsOutput;
public:
	book_reader(const char* fileName);
	book_reader(const std::string &fileName);
	book_reader(const std::string &&fileName);

	std::tuple<const std::vector<char>&, const std::vector<int>&> get_words(int align);
};

