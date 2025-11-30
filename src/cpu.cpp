#include "cpu.h"


template<int D>
void compute_clusters(const double* datapoints, double *clusters, int N, int K,
    unsigned char* assignments, int *delta)
{
    *delta = 0; // reset change counter
    for (int n = 0; n < N; n++)
    {
        unsigned char nearest_cluster;
        double min_distance = DBL_MAX;

        // for every point we find the nearest cluster based on Euclidean distance
        for(int k=0; k<K; ++k)
        {
            double distance = 0.0;

            #pragma unroll
            for(int d=0; d<D; ++d) 
            {
                double diff = datapoints[d * N + n] - clusters[k * D + d];
                distance += diff * diff;
            }

            if (distance < min_distance)
            {
                min_distance = distance;
                nearest_cluster = k;
            }
        }

        // if the assignment has changed, we update it and increment the change counter delta
        if (nearest_cluster != assignments[n])
        {
            assignments[n] = nearest_cluster;
            (*delta)++;
        }
    }    
}

template<int D>
void scatter_clusters(const double* datapoints, double *centroids, int N, int K, unsigned char* assignments, double *newCentroids, int *newCentroidsSize)
{
    for (int n = 0; n < N; n++)
    {
        unsigned char cluster_id = assignments[n];
        #pragma unroll
        // accumulate datapoint coordinates to the corresponding centroid
        for (int d = 0; d < D; d++)
        {
            newCentroids[cluster_id * D + d] += datapoints[d * N + n];
        }
        // increment the count of points assigned to this centroid
        newCentroidsSize[cluster_id]++;
    }

    // update centroids by averaging accumulated coordinates
    for (int k = 0; k < K; k++)
    {
        #pragma unroll
        for (int d = 0; d < D; d++)
        {
            // if no points assigned to this centroid, it remains unchanged
            if (newCentroidsSize[k] > 0)
                centroids[k * D + d] = newCentroids[k * D + d] / newCentroidsSize[k];

            newCentroids[k * D + d] = 0.0;
        }
        newCentroidsSize[k] = 0;
    }
}

template<int D>
void seq_kmeans(const double* datapoints, double* centroids, int N, int K, unsigned char* assignments, TimerManager* tm)
{
    TimerCPU timerCPU;
    tm->SetTimer(&timerCPU);

    int delta = N;
    double *newCentroids = (double*)malloc(K * D * sizeof(double));
    int *newCentroidsSize = (int*)malloc(K * sizeof(int));
    memset(newCentroids, 0, K * D * sizeof(double));
    memset(newCentroidsSize, 0, K * sizeof(int));
    memset(centroids, 0, K * D * sizeof(double));
    memset(assignments, 0, N * sizeof(unsigned char));
    // initialize centroids with the first K datapoints!
    for(int k=0; k<K; k++) {
        #pragma unroll
        for(int d=0; d<D; d++) {
            centroids[k * D + d] = datapoints[d * N + k];
        }
    }

    // main k-means iteration loop, if there are no changes in assignments (delta==0) or max iterations reached, we stop
    for (int iter = 0; iter < MAX_ITERATIONS && delta > 0; ++iter)
    {
        tm->Start();
        compute_clusters<D>(datapoints, centroids, N, K, assignments, &delta);
        tm->Stop();

        tm->Start();
        scatter_clusters<D>(datapoints, centroids, N, K, assignments, newCentroids, newCentroidsSize);
        tm->Stop();

        printf("Iteration: %d, changes: %d\n", iter, delta);

    }

    // visualization (for 3D data)
    auto visualizer = VisualizerFactory::create(ComputeType::CPU, D);
    if (visualizer && visualizer->canVisualize(D))
    {
        visualizer->visualize(datapoints, assignments, N, K, D);
    }

    free(newCentroids);
    free(newCentroidsSize);
}


// explicit template instantiation for seq_kmeans with D = 1 to 20
using KMeansFunc = void(const double*, double*, int, int, unsigned char*, TimerManager*);
template KMeansFunc seq_kmeans<1>;
template KMeansFunc seq_kmeans<2>;
template KMeansFunc seq_kmeans<3>;
template KMeansFunc seq_kmeans<4>;
template KMeansFunc seq_kmeans<5>;
template KMeansFunc seq_kmeans<6>;
template KMeansFunc seq_kmeans<7>;
template KMeansFunc seq_kmeans<8>;
template KMeansFunc seq_kmeans<9>;
template KMeansFunc seq_kmeans<10>;
template KMeansFunc seq_kmeans<11>;
template KMeansFunc seq_kmeans<12>;
template KMeansFunc seq_kmeans<13>;
template KMeansFunc seq_kmeans<14>;
template KMeansFunc seq_kmeans<15>;
template KMeansFunc seq_kmeans<16>;
template KMeansFunc seq_kmeans<17>;
template KMeansFunc seq_kmeans<18>;
template KMeansFunc seq_kmeans<19>;
template KMeansFunc seq_kmeans<20>;
