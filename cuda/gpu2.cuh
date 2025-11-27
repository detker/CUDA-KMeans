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


#define MAX_ITERATIONS 100

// Forward declaration
template<int D>
void thrust_kmeans_host(const double* datapoints, double* centroids,
    int N, int K, unsigned char* assignments, TimerManager *tm);