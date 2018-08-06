
#ifdef BOOST_ENABLE
#include <boost/test/unit_test.hpp>
#else
#include <iostream>
#include "search_words.cuh"
int main()
{
	find_words_tmp();
	std::cout << "Unit tests were not compiled." << std::endl
		<< "Before compiling unit tests, ensure that you have installed boost library." << std::endl
		<< "To compile unit tests, define symbol \"BOOST_ENABLE\"" << std::endl;
}

#endif
