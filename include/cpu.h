#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#include "timer.h"
#include "enum_types.h"
#include "visualizer.cuh"

#define MAX_ITERATIONS 100

// compute cluster assignments for each datapoint and count changes
template<int D>
void compute_clusters(const double* datapoints, double *clusters, int N, int K, unsigned char* assignments, int *delta);

// accumulate new centroids based on current assignments
template<int D>
void scatter_clusters(const double* datapoints, double *centroids, int N, int K, unsigned char* assignments, double *newCentroids, int *newCentroidsSize);

// sequential CPU k-means implementation (where D is a compile-time constant, D, K <= 20 and N <= 50mln)
// datapoints are in column-major format (SoA)
template<int D>
void seq_kmeans(const double* datapoints, double* centroids, int N, int K, unsigned char* assignments, TimerManager* tm);
