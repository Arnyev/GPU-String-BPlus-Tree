#include <cuda_runtime.h>
#include "functions.h"
#include "gpu_helper.cuh"
#include "sort_strings.cuh"
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>

using namespace thrust;

int* get_sorted_positions(int* d_positions, const int word_count, unsigned char* d_chars)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;
	cudaEventRecord(start);

	auto keys = device_malloc<ullong>(word_count);
	const auto destinations = device_malloc<int>(word_count);
	auto helper = device_malloc<int>(word_count);
	auto output = device_malloc<int>(word_count);

	create_consecutive_numbers(word_count, destinations);

	int offset = 0;
	int segment_size = 0;
	int current_count = word_count;

	while (true)
	{
		const int seg_chars = ceil(static_cast<double>(segment_size) / CHARBITS);
		create_hashes_with_seg(d_positions, d_chars, keys, helper, offset, segment_size, current_count, seg_chars);

		offset += CHARSTOHASH - seg_chars;

		sort_keys_and_positions(d_positions, keys, current_count);

		mark_singletons(d_positions, keys, destinations, helper, output, current_count);

		remove_handled_update_count(thrust::device_ptr<int>(d_positions), keys, destinations, helper, current_count);
		if (current_count == 0)
			break;
		std::cout << current_count << std::endl;

		flags_different_than_last(keys.get(), helper.get(), current_count);

		get_segments(helper, current_count);
		segment_size = compute_segment_size(helper.get(), current_count);
	}

	device_free(keys);
	device_free(destinations);
	device_free(helper);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "All sorting took " << milliseconds << " milliseconds" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return output.get();
}

sorting_output create_output(unsigned char* d_word_array, int* d_sorted_positions, int word_count)
{
	const device_ptr<int> sorted_positions(d_sorted_positions);

	const auto positions_end = remove_duplicates(d_sorted_positions, word_count, sorted_positions);

	word_count = positions_end - sorted_positions;

	const auto suffix_positions = device_malloc<int>(word_count + 1);

	get_suffix_positions(d_word_array, sorted_positions, positions_end, suffix_positions);

	int output_size;
	checkCudaErrors(cudaMemcpy(&output_size, suffix_positions.get() + word_count, sizeof(int), cudaMemcpyDeviceToHost));

	auto suffixes = device_malloc<uchar>(output_size);

	copy_suffixes(d_word_array, d_sorted_positions, word_count, suffix_positions, suffixes);

	const auto hashes = device_malloc<ullong>(word_count);
	create_hashes(d_word_array, sorted_positions, positions_end, hashes);

	const device_ptr<unsigned long long> hashes_end = get_unique_hashes(word_count, suffix_positions, hashes);

	const int hashes_count = hashes_end - hashes;

	return { hashes.get(), suffix_positions.get(), suffixes.get(), hashes_count, output_size };
}
