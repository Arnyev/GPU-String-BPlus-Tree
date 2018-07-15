#include "device_launch_parameters.h"
#include "parameters.h"

__device__ __host__ __inline__ ullong get_hash(uchar* words, const int chars_to_hash, const int my_position);
