#ifndef BOOST_ENABLE
#include <iostream>

int main()
{
	std::cout << "Unit tests were not compiled." << std::endl
		<< "Before compiling unit tests, ensure that you have installed boost library." << std::endl
		<< "To compile unit tests, define symbol \"BOOST_ENABLE\"" << std::endl;
}

#endif
