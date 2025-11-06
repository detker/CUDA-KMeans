#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

#define ERR(source) (perror(source), fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), exit(EXIT_FAILURE))
#define CUDA_CHECK(call) do {                                                                 \
    cudaError_t e = (call);                                                                   \
    if (e != cudaSuccess) {                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                                              \
    } } while(0)
#define MAX_ITERATIONS 100

void usage(char *name)
{
    fprintf(stderr, "USAGE: %s <data_file>\n", name);
    exit(EXIT_FAILURE);
}

typedef struct {
    float *datapoints;
    int K; // number of clusters
    long long N; // number of data points
    int D; // dimensionality
} Dataset;

__device__ __constant__ float DEVICE_INF = FLT_MAX;

__host__ __device__ inline void compute_distance_l2(const float *point1, const float *point2, int D, float *result)
{
    float sum = 0.0;
    for (int i = 0; i < D; i++) {
        float diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    *result = sum;
}

__global__ void compute_clusters(const float *datapoints, const float *centroids,
                                     int N, int K, int D,
                                     int *assignments, unsigned int *assignmentsChanged)
{
    extern __shared__ char sharedMem[];

    unsigned char *changedFlag = (unsigned char *)sharedMem;
    
    float *clusters = (float *)(sharedMem + sizeof(unsigned char)*blockDim.x);

    // for(int i=threadIdx.x; i<K; ++i)
    // {
        
    // }
    if (threadIdx.x < K)
    {
        for(int d=0; d<D; ++d)
        {
            clusters[threadIdx.x * D + d] = centroids[threadIdx.x * D + d];
        }
    }
    changedFlag[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float minDistance = DEVICE_INF;
    int bestCluster = 0;

    for (int k = 0; k < K; k++) {
        float distance;
        compute_distance_l2(&datapoints[idx * D], &clusters[k * D], D, &distance);
        if (distance < minDistance) {
            minDistance = distance;
            bestCluster = k;
        }
    }

    if (assignments[idx] != bestCluster)
    {
        assignments[idx] = bestCluster;
        changedFlag[threadIdx.x] = 1;
    }

    __syncthreads();

    for(int i=blockDim.x/2; i>0; i >>= 1)
    {
        if(threadIdx.x < i)
        {
            changedFlag[threadIdx.x] += changedFlag[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        assignmentsChanged[blockIdx.x] = changedFlag[0];
    }
}

__global__ void compute_delta(const unsigned int *assignmentsChanged, int N, unsigned int *delta)
{
    // tree based reduction here happens
    extern __shared__ unsigned int shm[];
    int tid = threadIdx.x;

    unsigned int sum = 0;
    for(int i=tid; i<N; i+=blockDim.x)
    {
        sum += assignmentsChanged[i];
    }
    shm[tid] = sum;
    __syncthreads();

    if(tid >= N) return;

    for(int i=blockDim.x/2; i>0; i>>=1)
    {
        if(tid < i)
        {
            shm[tid] += shm[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        *delta = shm[0];
    }
}

extern "C"
void kmeans_host(const float *datapoints, float *centroids,
                 int N, int K, int D, int *assignments)
{
    const float *deviceDatapoints;
    float *deviceCentroids;
    int *deviceAssignments;
    unsigned int *deviceAssignmentsChanged;
    float *newClusters;
    int *clustersSizes;
    size_t datapointsSize = N * D * sizeof(float);
    size_t centroidsSize = K * D * sizeof(float);
    size_t assignmentsSize = N * sizeof(int);
    size_t clustersSizesSize = K * sizeof(int);

    clustersSizes = (int*)malloc(clustersSizesSize);
    if (!clustersSizes) ERR("malloc clustersSizes failed.");
    memset((void*)clustersSizes, 0, clustersSizesSize);
    newClusters = (float*)malloc(centroidsSize);
    if (!newClusters) ERR("malloc newClusters failed.");
    memset((void*)newClusters, 0, centroidsSize);
    CUDA_CHECK(cudaMalloc((void**)&deviceDatapoints, datapointsSize));
    CUDA_CHECK(cudaMalloc((void**)&deviceCentroids, centroidsSize));
    CUDA_CHECK(cudaMalloc((void**)&deviceAssignments, assignmentsSize));

    // first we need to initialize centroids by first K datapoints (here we should ensure K <= N!)
    // TODO: add check for K <= N or simply memset centroids to 0 beforehand
    CUDA_CHECK(cudaMemcpy((void*)deviceDatapoints, (const void*)datapoints, datapointsSize, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy((void*)deviceCentroids, (const void*)datapoints, centroidsSize, cudaMemcpyHostToDevice));
    for(int k=0; k<K; ++k)
    {
        for(int d=0; d<D; ++d)
        {
            centroids[k * D + d] = datapoints[k * D + d];
        }
    }

    // Initialize assignments
    CUDA_CHECK(cudaMemset((void*)deviceAssignments, -1, assignmentsSize));
    
    unsigned delta = N;

    // Kernel launch parameters
    int threadsPerBlock = 128; // unsigned char limitation in shared memory usage
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = sizeof(unsigned char)*threadsPerBlock + sizeof(float)*K*D;
    //
    int threadsPerBlockDelta = 512;
    int blocksPerGridDelta = 1;
    int sharedMemSizeDelta = sizeof(unsigned int)*threadsPerBlockDelta;

    CUDA_CHECK(cudaMalloc((void**)&deviceAssignmentsChanged, sizeof(unsigned int)*blocksPerGrid));
    CUDA_CHECK(cudaMemset((void*)deviceAssignmentsChanged, 0, sizeof(unsigned int)*blocksPerGrid));

    for (int iter = 0; iter < MAX_ITERATIONS && delta > 0; iter++)
    {
        CUDA_CHECK(cudaMemcpy((void *)deviceCentroids, (const void *)centroids, centroidsSize, cudaMemcpyHostToDevice));
        // memset((void*)clustersSizes, 0, clustersSizesSize);

        compute_clusters<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(deviceDatapoints, deviceCentroids,
                                                    N, K, D, deviceAssignments, deviceAssignmentsChanged);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        

        // Reset delta
        delta = 0;
        unsigned int *deviceDelta;
        CUDA_CHECK(cudaMalloc((void**)&deviceDelta, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy((void*)deviceDelta, (const void*)&delta, sizeof(unsigned int), cudaMemcpyHostToDevice));

        compute_delta<<<blocksPerGridDelta, threadsPerBlockDelta, sharedMemSizeDelta>>>(deviceAssignmentsChanged, blocksPerGrid, deviceDelta);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy((void*)&delta, (const void*)deviceDelta, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree((void*)deviceDelta));

        // Here we should recompute centroids based on new assignments
        CUDA_CHECK(cudaMemset((void*)deviceAssignmentsChanged, 0, sizeof(unsigned int)*blocksPerGrid));
        CUDA_CHECK(cudaMemcpy((void*)assignments, (const void*)deviceAssignments, assignmentsSize, cudaMemcpyDeviceToHost));

        for(int i=0; i<N; ++i)
        {
            int idx = assignments[i];

            for(int d=0; d<D; ++d)
            {
                newClusters[idx * D + d] += datapoints[i * D + d];
            }
            clustersSizes[idx] += 1;
        }

        for(int k=0; k<K; ++k)
        {
            for(int d=0; d<D; ++d)
            {
                if(clustersSizes[k] > 0) centroids[k * D + d] = newClusters[k*D + d] / clustersSizes[k];
                else printf("kurwa");
                newClusters[k * D + d] = 0.0;
            }
            clustersSizes[k] = 0;
        }
    }

    // CUDA_CHECK(cudaMemcpy((void*)assignments, (const void*)deviceAssignments, assignmentsSize, cudaMemcpyDeviceToHost));
    //CUDA_CHECK(cudaMemcpy((void*)centroids, (const void*)deviceCentroids, centroidsSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree((void*)deviceDatapoints));
    CUDA_CHECK(cudaFree((void*)deviceCentroids));
    CUDA_CHECK(cudaFree((void*)deviceAssignments));
    CUDA_CHECK(cudaFree((void*)deviceAssignmentsChanged));
    free(clustersSizes);
    free(newClusters);
}

void load_data(Dataset *dataset, const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file) ERR("fopen failed.");

    if (fscanf(file, "%lld %d %d\n", &dataset->N, &dataset->D, &dataset->K) != 3)
    {
        ERR("fscanf failed.");
    }

    size_t datapointsSize = dataset->N * dataset->D * sizeof(float);
    dataset->datapoints = (float*)malloc(datapointsSize);
    if (!dataset->datapoints) ERR("malloc datapoints failed.");

    for (int i = 0; i < dataset->N; i++)
    {
        for (int d = 0; d < dataset->D; d++)
        {
            if (fscanf(file, "%f", &dataset->datapoints[i * dataset->D + d]) != 1)
            {
                ERR("fscanf datapoints failed.");
            }
        }
    }

    fclose(file);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        usage(argv[0]);
    }

    Dataset dataset;
    load_data(&dataset, argv[1]);

    for(int i=0; i<5; ++i)
    {
        for(int d=0; d<dataset.D; ++d)
        {
            printf("%f ", dataset.datapoints[i * dataset.D + d]);
        }
        printf("\n");
    }

    float *centroids = (float*)malloc(dataset.K * dataset.D * sizeof(float));
    int *assignments = (int*)malloc(dataset.N * sizeof(int));
    // points initialization bla bla bla
    kmeans_host(dataset.datapoints, centroids, dataset.N, dataset.K, dataset.D, assignments);

    // Print the results
    printf("Centroids:\n");
    for (int k = 0; k < dataset.K; k++) {
        for (int d = 0; d < dataset.D; d++) {
            printf("%.4f ", centroids[k * dataset.D + d]);
        }
        printf("\n");
    }

    free(dataset.datapoints);
    free(centroids);
    free(assignments);

    return EXIT_SUCCESS;
}