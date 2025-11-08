#pragma once

#include <stdio.h>
#include <cstdlib>

#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), exit(EXIT_FAILURE))

#define CUDA_CHECK(call) do {                                                                 \
    cudaError_t e = (call);                                                                   \
    if (e != cudaSuccess) {                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                                              \
    } } while(0)

void inline usage(char* name)
{
    fprintf(stderr, "USAGE: %s <data_format:txt|bin> <computation_method:cpu|gpu1|gpu2> <data_path> <output_path>\n", name);
    exit(EXIT_FAILURE);
}
