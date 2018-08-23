#include <chrono>
#include <iomanip>

#include "csv_logger.h"
#include <cuda_runtime_api.h>
#include <strstream>

#ifdef _DEBUG
const char *build = "DEBUG";
#else
const char *build = "RELEASE";
#endif

csv_logger::csv_logger(std::string & filePath) : file(filePath, std::ios::ate | std::ios::app)
{
	if (file.tellp() == std::strstream::pos_type(0))
	{
		file << "Date, Build, Algorithm, GPU, Build time[ms], Preproc time[ms], Exec time[ms], Postproc time[ms], Total time[ms], Dict size, Input size, Existing %\n";
	}
}

csv_logger::csv_logger(std::string && filePath) : csv_logger(filePath)
{
}

csv_logger::csv_logger(const char * filePath) : csv_logger(std::string(filePath))
{
}

csv_logger::~csv_logger()
{
	file.close();
}

void csv_logger::append(const char* algorithm, const float buildTime_ms, const float preprocTime_ms,
                        const float execTime_ms, const float postprocTime_ms, const size_t dictSize,
                        const size_t inputSize, const float exisiting)
{
	int device;
	cudaGetDevice(&device);
	struct cudaDeviceProp props {};
	cudaGetDeviceProperties(&props, device);
	const auto time_point = std::chrono::system_clock::now();
	const auto time = std::chrono::system_clock::to_time_t(time_point);
	struct tm timeinfo{};
	localtime_s(&timeinfo, &time);
	file << std::put_time(&timeinfo, "%c") << ','
		<< build << ','
		<< algorithm << ','
		<< props.name << ','
		<< buildTime_ms << ','
		<< preprocTime_ms << ','
		<< execTime_ms << ','
		<< postprocTime_ms << ','
		<< (preprocTime_ms + execTime_ms + postprocTime_ms) << ','
		<< dictSize << ','
		<< inputSize << ','
		<< exisiting << std::endl;
}

