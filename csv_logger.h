#pragma once
#include <string>
#include <fstream>

class csv_logger
{
	std::ofstream file;
public:
	csv_logger(std::string &filePath);
	csv_logger(std::string &&filePath);
	csv_logger(const char *filePath);
	~csv_logger();

	void append(const char* algorithm, const float buildTime_ms, const float preprocTime_ms, const float execTime_ms,
	            const float postprocTime_ms, const size_t dictSize, const size_t inputSize, const float exisiting);
};

