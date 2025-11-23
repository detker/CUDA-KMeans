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

__host__ inline unsigned int nextPowerOfTwo(unsigned int n);

__host__ __device__ inline void compute_distance_l2(const double* points, const double* clusters, int idx, int k_idx, int N, int K, int D, double* result);

__global__ void compute_clusters(const double* datapoints,
    int N, int K, int D,
    int* assignments, unsigned int* assignmentsChanged, double* newClusters, int* clustersSizes);

__global__ void compute_delta(const unsigned int* assignmentsChanged, int N, unsigned int* delta);

__global__ void scatter_clusters(const double* datapoints, const int* assignments,
    int N, int K, int D,
    double* newClusters, int* clustersSizes);

extern "C"
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, int D, int* assignments, TimerManager *tm);
