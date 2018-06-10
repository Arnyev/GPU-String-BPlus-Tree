#pragma once
#include <stdexcept>

class not_implemented : public std::logic_error
{
public:
	not_implemented() : std::logic_error("Function not yet implemented") { };
};
