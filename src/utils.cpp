#include "utils.h"


void load_txt_data(Dataset* dataset, const char* filename)
{
    FILE* file = fopen(filename, "r");
    if (!file) ERR("fopen failed.");

    if (fscanf(file, "%d %d %d\n", &dataset->N, &dataset->D, &dataset->K) != 3)
    {
        ERR("fscanf failed.");
    }

    size_t datapointsSize = dataset->N * dataset->D * sizeof(double);

    dataset->datapoints = (double*)malloc(datapointsSize);
    if (!dataset->datapoints) ERR("malloc datapoints failed.");

    for (int i = 0; i < dataset->N; i++)
    {
        for (int d = 0; d < dataset->D; d++)
        {
            if (fscanf(file, "%lf", &dataset->datapoints[d * dataset->N + i]) != 1)
            {
                ERR("fscanf datapoints failed.");
            }
        }
    }

    fclose(file);
}

void load_bin_data(Dataset* dataset, const char* filename)
{
    FILE* file = fopen(filename, "rb");
    if (!file) ERR("fopen failed.");

    if (fread(&dataset->N, sizeof(int), 1, file) != 1 ||
        fread(&dataset->D, sizeof(int), 1, file) != 1 ||
        fread(&dataset->K, sizeof(int), 1, file) != 1)
    {
        ERR("fread header failed.");
    }

    int blocks = (dataset->N + 1023) / 1024;
    double *chunkBuffer = (double*)malloc(1024 * dataset->D * sizeof(double));
    if (!chunkBuffer) ERR("malloc chunkBuffer failed.");

    size_t datapointsSize = dataset->N * dataset->D * sizeof(double);
    dataset->datapoints = (double*)malloc(datapointsSize);
    if (!dataset->datapoints) ERR("malloc datapoints failed.");

    for (int b = 0; b < blocks; ++b)
    {
        int pointsInThisBlock = ((b == blocks - 1) && (dataset->N % 1024 != 0)) ? (dataset->N % 1024) : 1024;
        if (fread(chunkBuffer, sizeof(double), pointsInThisBlock * dataset->D, file) != pointsInThisBlock * dataset->D)
        {
            ERR("fread datapoints chunk failed.");
        }
        for (int i = 0; i < pointsInThisBlock; ++i)
        {
            for (int d = 0; d < dataset->D; ++d)
            {
                dataset->datapoints[d * dataset->N + (b * 1024 + i)] = chunkBuffer[i * dataset->D + d];
                // direct SoA load
            }
        }
    }

    free(chunkBuffer);
    fclose(file);
}

void save_output(double** centroids, unsigned char** assignments, Dataset* dataset, char* output_path)
{
    FILE* file = fopen(output_path, "w");
    if (!file) ERR("fopen failed.");

    for (int k = 0; k < dataset->K; k++) {
        for (int d = 0; d < dataset->D; d++) {
            fprintf(file, "%.4lf ", (*centroids)[k * dataset->D + d]);
        }
        fprintf(file, "\n");
    }

    for (int i = 0; i < dataset->N; i++) {
        fprintf(file, "%d\n", (*assignments)[i]);
    }

    fclose(file);
}

void parse_args(int argc, char** argv, DataType* data_format, ComputeType* compute_method, char** data_path, char** output_path)
{
    if (argc != 5) {
        usage(argv[0]);
    }

    if (strcmp(argv[1], "txt") == 0)
    {
        *data_format = DataType::TXT;
    }
    else if (strcmp(argv[1], "bin") == 0)
    {
        *data_format = DataType::BIN;
    }
    else usage(argv[0]);

    if (strcmp(argv[2], "gpu1") == 0)
    {
        *compute_method = ComputeType::GPU1;
    }
    else if (strcmp(argv[2], "gpu2") == 0)
    {
        *compute_method = ComputeType::GPU2;
    }
    else if (strcmp(argv[2], "cpu") == 0)
    {
        *compute_method = ComputeType::CPU;
    }
    else usage(argv[0]);

    *data_path = argv[3];

    *output_path = argv[4];
}

template<typename T>
void col_to_row_major(const T *col_major, T *row_major, int N, int D)
{
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            row_major[n * D + d] = col_major[d * N + n];
        }
    }
}

template void col_to_row_major<double>(const double *col_major, double *row_major, int N, int D);
template void col_to_row_major<int>(const int *col_major, int *row_major, int N, int D);
