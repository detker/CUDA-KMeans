#pragma once

#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdint>

#include "viz.cuh"
#include "error_utils.h"
#include "timer.h"
#include "utils.h"
#include "visualizer.cuh"
#include "enum_types.h"

#define MAX_ITERATIONS 100

#ifdef __CUDACC__
__device__ __constant__ double DEVICE_INF = DBL_MAX;
#endif

// parallel GPU k-means implementation with only custom kernels (where D is a compile-time constant, D, K <= 20 and N <= 50mln)
// datapoints are in column-major format (SoA)
template<int D>
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, unsigned char* assignments, TimerManager *tm);

// compute cluster assignments for each datapoint and count changes
template<int D>
__global__ void compute_clusters(const double* datapoints, double *centroids,
    int N, int K,
    unsigned char* assignments, unsigned int* assignmentsChanged, double* newClusters, int* clustersSizes);

// accumulate new centroids based on current assignments
template<int D>
__global__ void scatter_clusters(const double* datapoints, const unsigned char* assignments,
    int N, int K,
    double* newClusters, int* clustersSizes);
