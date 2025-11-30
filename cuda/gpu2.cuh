#pragma once

#include <float.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>

#include "viz.cuh"
#include "timer.h"
#include "utils.h"
#include "visualizer.cuh"
#include "enum_types.h"


#define MAX_ITERATIONS 100

// parallel GPU k-means implementation using thrust (where D is a compile-time constant, D, K <= 20 and N <= 50mln)
// datapoints are in column-major format (SoA)
template<int D>
void thrust_kmeans_host(const double* datapoints, double* centroids,
    int N, int K, unsigned char* assignments, TimerManager *tm);