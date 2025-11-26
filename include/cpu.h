#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#include "timer.h"

#define MAX_ITERATIONS 100

template<int D>
double compute_distance_l2(const double* point1, const double* point2);

template<int D>
void compute_clusters(const double* datapoints, double *clusters, int N, int K, int* assignments, int *delta);

template<int D>
void scatter_clusters(const double* datapoints, double *centroids, int N, int K, int* assignments, double *newCentroids, int *newCentroidsSize);

template<int D>
void seq_kmeans(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager* tm);