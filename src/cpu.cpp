#include "cpu.h"


double compute_distance_l2(const double* point1, const double* point2, int D)
{
    double sum = 0.0;
    for (int d = 0; d < D; d++)
    {
        double diff = point1[d] - point2[d];
        sum += diff * diff;
    }
    return sum;
}


void compute_clusters(const double* datapoints, double *clusters, int N, int K, int D,
    int* assignments, int *delta)
{
    *delta = 0;
    for (int n = 0; n < N; n++)
    {
        const double *current_point = datapoints + n * D;
        int nearest_cluster = -1;
        double min_distance = DBL_MAX;

        for(int k=0; k<K; ++k)
        {
            const double *current_cluster = clusters + k * D;
            double distance = compute_distance_l2(current_point, current_cluster, D);
            if (distance < min_distance)
            {
                min_distance = distance;
                nearest_cluster = k;
            }
        }

        if (nearest_cluster != assignments[n])
        {
            assignments[n] = nearest_cluster;
            (*delta)++;
        }
    }    
}

void scatter_clusters(const double* datapoints, double *centroids, int N, int K, int D, int* assignments, double *newCentroids, int *newCentroidsSize)
{
    for (int n = 0; n < N; n++)
    {
        int cluster_id = assignments[n];
        const double *current_point = datapoints + n * D;
        for (int d = 0; d < D; d++)
        {
            newCentroids[cluster_id * D + d] += current_point[d];
        }
        newCentroidsSize[cluster_id]++;
    }

    for (int k = 0; k < K; k++)
    {
        for (int d = 0; d < D; d++)
        {
            if (newCentroidsSize[k] > 0)
                centroids[k * D + d] = newCentroids[k * D + d] / newCentroidsSize[k];
            // else
            //     centroids[k * D + d] = 0.0;

            newCentroids[k * D + d] = 0.0;
        }
        newCentroidsSize[k] = 0;
    }
}

void seq_kmeans(const double* datapoints, double* centroids, int N, int K, int D, int* assignments, TimerManager* tm)
{



    TimerCPU timerCPU;
    tm->SetTimer(&timerCPU);

    int delta = N;
    double *newCentroids = (double*)malloc(K * D * sizeof(double));
    int *newCentroidsSize = (int*)malloc(K * sizeof(int));
    memset(newCentroids, 0, K * D * sizeof(double));
    memset(newCentroidsSize, 0, K * sizeof(int));
    memset(centroids, 0, K * D * sizeof(double));
    for(int k=0; k<K; k++) {
        // if(k >= N) break;
        for(int d=0; d<D; d++) {
            centroids[k * D + d] = datapoints[k * D + d];
        }
    }

    printf("CPU centroid 0: ");
    for (int d = 0; d < D; d++) {
        printf("%.10f ", centroids[0 * D + d]);
    }
    printf("\n");
    for (int iter = 0; iter < MAX_ITERATIONS && delta > 0; ++iter)
    {
        tm->Start();
        compute_clusters(datapoints, centroids, N, K, D, assignments, &delta);
        tm->Stop();

        tm->Start();
        scatter_clusters(datapoints, centroids, N, K, D, assignments, newCentroids, newCentroidsSize);
        tm->Stop();

        printf("Iteration: %d, changes: %d\n", iter, delta);

    }

    free(newCentroids);
    free(newCentroidsSize);
}
