#define MAX_ITERATIONS 100

void thrust_kmeans_host(double* datapoints, double* centroids,
    int N, int K, int D, int* assignments);