#include <cstdlib>
#include <stdio.h>
#include "gpu1.cuh"
#include "gpu2.cuh"

#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), exit(EXIT_FAILURE))
#define CPU 0
#define GPU1 1
#define GPU2 2
#define TXT_DATA_FORMAT 0
#define BINARY_DATA_FORMAT 1


void usage(char* name)
{
    fprintf(stderr, "USAGE: %s <data_format:txt|bin> <computation_method:cpu|gpu1|gpu2> <data_path> <output_path>\n", name);
    exit(EXIT_FAILURE);
}

typedef struct {
    double* datapoints;
    int N; // number of data points
    int D; // dimensionality
    int K; // number of clusters
} Dataset;


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

void parse_args(int argc, char **argv, unsigned char *data_format, unsigned char *compute_method, char** data_path, char** output_path)
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


int main(int argc, char** argv)
{
	unsigned char data_format, compute_method;
    char* data_path, *output_path;
    parse_args(argc, argv, &data_format, &compute_method, &data_path, &output_path);

    Dataset dataset;

    if (data_format == TXT_DATA_FORMAT) load_txt_data(&dataset, data_path);
    else load_bin_data(&dataset, data_path);

    for (int i = 0; i < 5; ++i)
    {
        for (int d = 0; d < dataset.D; ++d)
        {
            printf("%lf ", dataset.datapoints[i * dataset.D + d]);
        }
        printf("\n");
    }

    double* centroids = (double*)malloc(dataset.K * dataset.D * sizeof(double));
    int* assignments = (int*)malloc(dataset.N * sizeof(int));
	if (!centroids || !assignments) ERR("malloc centroids/assignments failed.");

    if (compute_method == GPU1)
        kmeans_host(dataset.datapoints, centroids, dataset.N, dataset.K, dataset.D, assignments);
    else if (compute_method == GPU2)
    {
		thrust_kmeans_host(dataset.datapoints, centroids, dataset.N, dataset.K, dataset.D, assignments);
    }
    else // CPU
    {
    }

    // Print the results
    printf("Centroids:\n");
    for (int k = 0; k < dataset.K; k++) {
        for (int d = 0; d < dataset.D; d++) {
            printf("%.4lf ", centroids[k * dataset.D + d]);
        }
        printf("\n");
    }
    save_output(&centroids, &assignments, &dataset, output_path);

    free(dataset.datapoints);
    free(centroids);
    free(assignments);

    return EXIT_SUCCESS;
}