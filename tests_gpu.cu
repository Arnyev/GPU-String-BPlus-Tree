#include "preparing_tree.cuh"
#include "helpers.h"

using namespace std;
using namespace thrust;

bool test_output(const host_vector<char> & words, const host_vector<int> & positions_host, const device_vector<char> & words_device, device_vector<int> & sorted_positions)
{
	device_vector<ullong> hashes;
	device_vector<int> positions;
	device_vector<char> suffixes;

	create_output(words_device, sorted_positions, hashes, positions, suffixes);
	vector<ullong> hashes_cpu;
	vector<int> positions_cpu;
	vector<char> suffixes_cpu;

	create_output_cpu(words, positions_host, hashes_cpu, positions_cpu, suffixes_cpu);

	if (hashes.size() != hashes_cpu.size())
	{
		cout << "Bad hashes count" << endl;
		return false;
	}

	auto gpu_hashes = from_vector_dev(hashes);
	for (size_t i = 0; i < gpu_hashes.size(); i++)
		if (gpu_hashes[i] != hashes_cpu[i])
		{
			cout << "Bad hash" << endl;
			return false;
		}

	if (suffixes.size() != suffixes_cpu.size())
	{
		cout << "Bad suffix size" << endl;
		return false;
	}

	auto gpu_suffixes = from_vector_dev(suffixes);
	for (size_t i = 0; i < gpu_suffixes.size(); i++)
	{
		if (gpu_suffixes[i] != suffixes_cpu[i])
		{
			cout << "Bad char" << endl;
			return false;
		}
	}

	auto gpu_positions = from_vector_dev(positions);
	for (size_t i = 0; i < gpu_positions.size(); i++)
		if (gpu_positions[i] != positions_cpu[i])
		{
			cout << "Bad position" << endl;
			return false;
		}

	return true;
}
