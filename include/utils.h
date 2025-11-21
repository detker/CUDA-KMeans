#pragma once

#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <math.h>

#include "error_utils.h"

#define CPU 0
#define GPU1 1
#define GPU2 2
#define TXT_DATA_FORMAT 0
#define BINARY_DATA_FORMAT 1

typedef struct {
    //std::vector<double> datapoints;
	double* datapoints; // pointer to data points
    int N; // number of data points
    int D; // dimensionality
    int K; // number of clusters
} Dataset;

void usage(char* name);

void load_txt_data(Dataset* dataset, const char* filename);

void load_bin_data(Dataset* dataset, const char* filename);

void save_output(double** centroids, int** assignments, Dataset* dataset, char* output_path);

void parse_args(int argc, char** argv, unsigned char* data_format, unsigned char* compute_method, char** data_path, char** output_path);

template<typename T>
void row_to_col_major(const T *row_major, T *col_major, int N, int D);

void compute_bounds(const double *pts, int N, float &minx, float &maxx, float &miny, float &maxy, float &minz, float &maxz);