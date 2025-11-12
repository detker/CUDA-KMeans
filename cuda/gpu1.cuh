#pragma once

#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <float.h>

#include "viz.cuh"
#include "error_utils.h"
#include "timer.h"


#define MAX_ITERATIONS 100

__host__ __device__ inline void compute_distance_l2(const double* point1, const double* point2, int D, double* result);

// __global__ void compute_clusters(const double* datapoints, const double* centroids,
//     int N, int K, int D,
//     int* assignments, unsigned int* assignmentsChanged);

__global__ void compute_clusters(const double* datapoints,
    int N, int K, int D,
    int* assignments, unsigned int* assignmentsChanged, double* newClusters, int* clustersSizes);

__global__ void compute_delta(const unsigned int* assignmentsChanged, int N, unsigned int* delta);

extern "C"
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, int D, int* assignments, TimerManager *tm);
