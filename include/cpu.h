#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#include "timer.h"

#define MAX_ITERATIONS 100

double compute_distance_l2(const double* point1, const double* point2, int D);

void compute_clusters(const double* datapoints, double *clusters, int N, int K, int D, int* assignments, int *delta);

void scatter_clusters(const double* datapoints, double *centroids, int N, int K, int D, int* assignments, double *newCentroids, int *newCentroidsSize);

void seq_kmeans(const double* datapoints, double* centroids, int N, int K, int D, int* assignments, TimerManager* tm);