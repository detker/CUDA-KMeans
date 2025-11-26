#pragma once

#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <float.h>

#include "viz.cuh"
#include "error_utils.h"
#include "timer.h"
#include "utils.h"


#define MAX_ITERATIONS 100

// Device constant
#ifdef __CUDACC__
__device__ __constant__ double DEVICE_INF = DBL_MAX;
#endif

// Forward declaration
template<int D>
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, int* assignments, TimerManager *tm);
