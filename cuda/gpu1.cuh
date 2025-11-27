#pragma once

#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <float.h>

#include "viz.cuh"
#include "error_utils.h"
#include "timer.h"
#include "utils.h"
#include "visualizer.cuh"


#define MAX_ITERATIONS 100

// Device constant
#ifdef __CUDACC__
__device__ __constant__ double DEVICE_INF = DBL_MAX;
#endif

// Forward declaration
template<int D>
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, unsigned char* assignments, TimerManager *tm);

template<int D>
__global__ void update_centroids(double* centroids, const double* newClusters, 
                                  const int* clustersSizes, int K);

template<int D>
__global__ void compute_clusters(const double* datapoints, double *centroids,
    int N, int K,
    unsigned char* assignments, unsigned int* assignmentsChanged, double* newClusters, int* clustersSizes);

template<int D>
__global__ void scatter_clusters(const double* datapoints, const unsigned char* assignments,
    int N, int K,
    double* newClusters, int* clustersSizes, double* centroids);
