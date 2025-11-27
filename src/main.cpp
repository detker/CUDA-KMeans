#include <stdio.h>
#include <cstdlib>

// #include "gpu1.cuh"
// #include "gpu2.cuh"
// #include "cpu.h"
#include "utils.h"
#include "error_utils.h"
#include "timer.h"
#include "dispatcher.h"
#include "enum_types.h"

void print_info(Dataset* d, DataType data_format)
{
    printf("Data format: %s\n", (data_format == DataType::BIN) ? "binary" : "text");
    printf("Number of data points: %d\n", d->N);
    printf("Dimensionality: %d\n", d->D);
	printf("Number of clusters: %d\n", d->K);
}

int main(int argc, char** argv)
{
    TimerCPU timerCPU;
    TimerManager tm;
	tm.SetTimer(&timerCPU);

	// unsigned char data_format, compute_method;
    DataType data_format;
    ComputeType compute_method;
    char* data_path, *output_path;
    Dataset dataset;
    
    parse_args(argc, argv, &data_format, &compute_method, &data_path, &output_path);

    printf("Reading data to CPU...\n");
    tm.Start();
    if (data_format == DataType::TXT) load_txt_data(&dataset, data_path);
    else load_bin_data(&dataset, data_path);
    tm.Stop();
    print_info(&dataset, data_format);
    float time_load_data = tm.TotalElapsedSeconds();
	printf("Data read time: %.4f seconds\n", time_load_data);

    //for (int i = 0; i < 5; ++i)
    //{
    //    for (int d = 0; d < dataset.D; ++d)
    //    {
    //        printf("%lf ", dataset.datapoints[i * dataset.D + d]);
    //    }
    //    printf("\n");
    //}

    double* centroids = (double*)malloc(dataset.K * dataset.D * sizeof(double));
    unsigned char* assignments = (unsigned char*)malloc(dataset.N * sizeof(unsigned char));
	if (!centroids || !assignments) ERR("malloc centroids/assignments failed.");

    // if (compute_method == GPU1)
    // {
    //     printf("Running K-means on gpu1 (parallel, custom kernels)...\n");
    //     kmeans_host(dataset.datapoints, centroids, dataset.N, dataset.K, dataset.D, assignments, &tm);
    // }
    // else if (compute_method == GPU2)
    // {
    //     printf("Running K-means on gpu2 (parallel, thrust)...\n");
	// 	thrust_kmeans_host(dataset.datapoints, centroids, dataset.N, dataset.K, dataset.D, assignments, &tm);
    // }
    // else // CPU
    // {
    //     printf("Running K-means on CPU (sequential)...\n");
    //     seq_kmeans(dataset.datapoints, centroids, dataset.N, dataset.K, dataset.D, assignments, &tm);
    // }

    RunKMeans(dataset.datapoints, centroids, assignments, dataset.N, dataset.K, dataset.D, compute_method, &tm);
    // RunKMeans(dataset, centroids, assignments, compute_method, &tm);

    float time_computing = tm.TotalElapsedSeconds() - time_load_data;
	printf("K-means computation time: %.4f seconds\n", time_computing);

    // Print the results
    //printf("Centroids:\n");
    //for (int k = 0; k < dataset.K; k++) {
    //    for (int d = 0; d < dataset.D; d++) {
    //        printf("%.4lf ", centroids[k * dataset.D + d]);
    //    }
    //    printf("\n");
    //}

    printf("Saving results to file...\n");
	tm.SetTimer(&timerCPU);
	tm.Start();
    save_output(&centroids, &assignments, &dataset, output_path);
	tm.Stop();
    float time_save_data = tm.TotalElapsedSeconds() - time_computing - time_load_data;
	printf("Results saved, time: %.4f seconds\n", time_save_data);

    free(dataset.datapoints);
    free(centroids);
    free(assignments);

    printf("Total execution time: %.4f seconds\n", tm.TotalElapsedSeconds());

    return EXIT_SUCCESS;
}
