#include <functional>
#include <memory>
#ifdef BOOST_ENABLE
#include <boost/test/unit_test.hpp>
#else
#include <iostream>
#include "search_words.cuh"

class test_case_abstract_t
{
public:
	std::string m_name;
	test_case_abstract_t(std::string &name) : m_name(name)
	{
		
	}
	virtual bool operator()() = 0;
};

template<typename Functor>
class test_case_t : public test_case_abstract_t
{
	Functor m_functor;
public:
	test_case_t(Functor functor, std::string &name) : test_case_abstract_t(name), m_functor(functor)
	{
		
	}
	virtual bool operator()()
	{
		return m_functor();
	}
};

template<typename Functor>
std::shared_ptr<test_case_abstract_t> make_test_case(Functor functor, std::string &&name)
{
	return std::make_shared<test_case_t<Functor>>(functor, name);
}

int main()
{
	std::srand(0x80085);
	auto tests = {
		make_test_case(less_than_14_chars_1, "Less than 14 chars, 01"),
		make_test_case(less_than_14_chars_2, "Less than 14 chars, 02"),
		make_test_case(repeating_prefixes_1, "Repeating prefixes, 01"),
		make_test_case(repeating_prefixes_2, "Repeating prefixes, 02"),
		make_test_case(great_test_t(50000, 4, 15, 10000, { 'a', 'b' }, 0x80085), "Great test, 01"),
		make_test_case(great_test_t(200000, 4, 15, 50000, { 'a', 'b', 'c', 'd' }, 0xDD05), "Great test, 02"),
	};
	for (auto && test_p : tests)
	{
		test_case_abstract_t &test = *test_p;
		std::cout << test.m_name << "| ";
		if (test())
			std::cout << "   Ok\n";
		else
			std::cout << "Wrong\n";
	}
}

#endif
