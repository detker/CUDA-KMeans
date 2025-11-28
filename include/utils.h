#pragma once

#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <math.h>

#include "error_utils.h"
#include "enum_types.h"

typedef struct {
	double* datapoints;
    int N;
    int D;
    int K;
} Dataset;

// load dataset from a binary file directly into SoA format using chunked reading
void load_bin_data(Dataset* dataset, const char* filename);

// load dataset from a text file directly into SoA format
void load_txt_data(Dataset* dataset, const char* filename);

// save centroids and assignments to a text file
void save_output(double** centroids, unsigned char** assignments, Dataset* dataset, char* output_path);

void parse_args(int argc, char** argv, DataType *data_format, ComputeType* compute_method, char** data_path, char** output_path);

template<typename T>
void col_to_row_major(const T *col_major, T *row_major, int N, int D);
