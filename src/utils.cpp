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

//    dataset->D = 3;
//    dataset->K = 10;

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

template<typename T>
void row_to_col_major(const T *row_major, T *col_major, int N, int D)
{
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            col_major[d * N + n] = row_major[n * D + d];
        }
    }
}

template void row_to_col_major<double>(const double *row_major, double *col_major, int N, int D);
template void row_to_col_major<int>(const int *row_major, int *col_major, int N, int D);

void compute_bounds(const double *pts, int N, float &minx, float &maxx, float &miny, float &maxy, float &minz, float &maxz)
{
    maxx = (float)pts[0];
    miny = (float)pts[1];
    maxy = (float)pts[1];
    minz = (float)pts[2];
    maxz = (float)pts[2];
    for (size_t i = 1;i < N;++i)
    {
        minx = fminf(minx, (float)pts[i * 3]); maxx = fmaxf(maxx, (float)pts[i * 3]);
        miny = fminf(miny, (float)pts[i * 3 + 1]); maxy = fmaxf(maxy, (float)pts[i * 3 + 1]);
        minz = fminf(minz, (float)pts[i * 3 + 2]); maxz = fmaxf(maxz, (float)pts[i * 3 + 2]);
    }

    float px = fmaxf(1e-6f, (maxx - minx) * 0.01f);
    float py = fmaxf(1e-6f, (maxy - miny) * 0.01f);
    float pz = fmaxf(1e-6f, (maxz - minz) * 0.01f);
    minx -= px; maxx += px;
    miny -= py; maxy += py;
    minz -= pz; maxz += pz;
}