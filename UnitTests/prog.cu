#include <functional>
#ifdef BOOST_ENABLE
#include <boost/test/unit_test.hpp>
#else
#include <iostream>
#include "search_words.cuh"

using test_function = std::function<bool()>;
using test_case = std::tuple<test_function, std::string>;

int main()
{
	auto tests = {
		test_case(less_than_14_chars_1, "Less than 14 chars, 01"),
		test_case(less_than_14_chars_2, "Less than 14 chars, 02"),
		test_case(repeating_prefixes_1, "Repeating prefixes, 01"),
		test_case(repeating_prefixes_2, "Repeating prefixes, 02"),
	};
	test_function fun;
	std::string desc;
	for (auto && test : tests)
	{
		std::tie(fun, desc) = test;
		std::cout << desc << "| ";
		if (fun())
			std::cout << "   Ok\n";
		else
			std::cout << "Wrong\n";
	}
}

#endif
