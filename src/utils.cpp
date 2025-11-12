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
    //dataset->datapoints.resize(dataset->N * dataset->D);

    dataset->datapoints = (double*)malloc(datapointsSize);
    if (!dataset->datapoints) ERR("malloc datapoints failed.");

    for (int i = 0; i < dataset->N; i++)
    {
        for (int d = 0; d < dataset->D; d++)
        {
            if (fscanf(file, "%lf", &dataset->datapoints[i * dataset->D + d]) != 1)
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

    // dataset->D = 3;
    // dataset->K = 10;

	//dataset->datapoints.resize(dataset->N * dataset->D);

    size_t datapointsSize = dataset->N * dataset->D * sizeof(double);
    dataset->datapoints = (double*)malloc(datapointsSize);
    if (!dataset->datapoints) ERR("malloc datapoints failed.");
    if (fread(dataset->datapoints, sizeof(double), dataset->N * dataset->D, file) != dataset->N * dataset->D)
    {
        ERR("fread datapoints failed.");
    }

    fclose(file);
}

void save_output(double** centroids, int** assignments, Dataset* dataset, char* output_path)
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

void parse_args(int argc, char** argv, unsigned char* data_format, unsigned char* compute_method, char** data_path, char** output_path)
{
    if (argc != 5) {
        usage(argv[0]);
    }

    if (strcmp(argv[1], "txt") == 0)
    {
        *data_format = TXT_DATA_FORMAT;
    }
    else if (strcmp(argv[1], "bin") == 0)
    {
        *data_format = BINARY_DATA_FORMAT;
    }
    else usage(argv[0]);

    if (strcmp(argv[2], "gpu1") == 0)
    {
        *compute_method = GPU1;
    }
    else if (strcmp(argv[2], "gpu2") == 0)
    {
        *compute_method = GPU2;
    }
    else if (strcmp(argv[2], "cpu") == 0)
    {
        *compute_method = CPU;
    }
    else usage(argv[0]);

    *data_path = argv[3];

    *output_path = argv[4];
}
