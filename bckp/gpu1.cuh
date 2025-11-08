#pragma once
#include <cuda_runtime.h>
#include <float.h>

#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), exit(EXIT_FAILURE))
#define CUDA_CHECK(call) do {                                                                 \
    cudaError_t e = (call);                                                                   \
    if (e != cudaSuccess) {                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                                              \
    } } while(0)
#define MAX_ITERATIONS 100

__device__ __constant__ double DEVICE_INF = DBL_MAX;

__host__ __device__ inline void compute_distance_l2(const double* point1, const double* point2, int D, double* result);

__global__ void compute_clusters(const double* datapoints, const double* centroids,
    int N, int K, int D,
    int* assignments, unsigned int* assignmentsChanged);

__global__ void compute_delta(const unsigned int* assignmentsChanged, int N, unsigned int* delta);

extern "C"
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, int D, int* assignments);
