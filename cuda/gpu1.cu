#include "gpu1.cuh"

__device__ __constant__ double DEVICE_INF = DBL_MAX;

__host__ __device__ inline void compute_distance_l2(const double* point1, const double* point2, int D, double* result)
{
    double sum = 0.0;
    for (int i = 0; i < D; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    *result = sum;
}

__global__ void compute_clusters(const double* datapoints, const double* centroids,
    int N, int K, int D,
    int* assignments, unsigned int* assignmentsChanged)
{
    extern __shared__ char sharedMem[];

    unsigned char* changedFlag = (unsigned char*)sharedMem;

    double* clusters = (double*)(sharedMem + sizeof(unsigned char) * blockDim.x);

    // for(int i=threadIdx.x; i<K; ++i)
    // {

    // }
    if (threadIdx.x < K)
    {
        for (int d = 0; d < D; ++d)
        {
            clusters[threadIdx.x * D + d] = centroids[threadIdx.x * D + d];
        }
    }
    changedFlag[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
    {
        sharedMem[threadIdx.x] = 0;
        return;
    }

    double minDistance = DEVICE_INF;
    int bestCluster = 0;

    for (int k = 0; k < K; k++) {
        double distance;
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

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            changedFlag[threadIdx.x] += changedFlag[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        assignmentsChanged[blockIdx.x] = changedFlag[0];
    }
	sharedMem[threadIdx.x] = 0;
}

__global__ void compute_delta(const unsigned int* assignmentsChanged, int N, unsigned int* delta)
{
    // tree based reduction here happens
    extern __shared__ unsigned int shm[];
    int tid = threadIdx.x;

    unsigned int sum = 0;
    for (int i = tid; i < N; i += blockDim.x)
    {
        sum += assignmentsChanged[i];
    }
    shm[tid] = sum;
    __syncthreads();

    if (tid >= N)
    {
        shm[tid] = 0;
        return;
    }

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (tid < i)
        {
            shm[tid] += shm[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        *delta = shm[0];
    }
    shm[tid] = 0;
}

extern "C"
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, int D, int* assignments, TimerManager *tm)
{
    TimerGPU timerGPU;
    tm->SetTimer(&timerGPU);

    const double* deviceDatapoints;
    double* deviceCentroids;
    int* deviceAssignments;
    unsigned int* deviceAssignmentsChanged;
    double* newClusters;
    int* clustersSizes;
    size_t datapointsSize = N * D * sizeof(double);
    size_t centroidsSize = K * D * sizeof(double);
    size_t assignmentsSize = N * sizeof(int);
    size_t clustersSizesSize = K * sizeof(int);

    clustersSizes = (int*)malloc(clustersSizesSize);
    if (!clustersSizes) ERR("malloc clustersSizes failed.");
    memset((void*)clustersSizes, 0, clustersSizesSize);
    newClusters = (double*)malloc(centroidsSize);
    if (!newClusters) ERR("malloc newClusters failed.");
    memset((void*)newClusters, 0, centroidsSize);
    CUDA_CHECK(cudaMalloc((void**)&deviceDatapoints, datapointsSize));
    CUDA_CHECK(cudaMalloc((void**)&deviceCentroids, centroidsSize));
    CUDA_CHECK(cudaMalloc((void**)&deviceAssignments, assignmentsSize));

    // first we need to initialize centroids by first K datapoints (here we should ensure K <= N!)
    // TODO: add check for K <= N or simply memset centroids to 0 beforehand
    CUDA_CHECK(cudaMemcpy((void*)deviceDatapoints, (const void*)datapoints, datapointsSize, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy((void*)deviceCentroids, (const void*)datapoints, centroidsSize, cudaMemcpyHostToDevice));
    for (int k = 0; k < K; ++k)
    {
        for (int d = 0; d < D; ++d)
        {
            centroids[k * D + d] = datapoints[k * D + d];
        }
    }

    // Initialize assignments
    CUDA_CHECK(cudaMemset((void*)deviceAssignments, 0, assignmentsSize));

    unsigned delta = N;

    // Kernel launch parameters
    int threadsPerBlock = 128; // unsigned char limitation in shared memory usage
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = sizeof(unsigned char) * threadsPerBlock + sizeof(double) * K * D;
    //
    int threadsPerBlockDelta = 512;
    int blocksPerGridDelta = 1;
    int sharedMemSizeDelta = sizeof(unsigned int) * threadsPerBlockDelta;

    CUDA_CHECK(cudaMalloc((void**)&deviceAssignmentsChanged, sizeof(unsigned int) * blocksPerGrid));
    CUDA_CHECK(cudaMemset((void*)deviceAssignmentsChanged, 0, sizeof(unsigned int) * blocksPerGrid));

    for (int iter = 0; iter < MAX_ITERATIONS && delta > 0; iter++)
    {
        CUDA_CHECK(cudaMemcpy((void*)deviceCentroids, (const void*)centroids, centroidsSize, cudaMemcpyHostToDevice));
        // memset((void*)clustersSizes, 0, clustersSizesSize);

        tm->Start();
        compute_clusters << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (deviceDatapoints, deviceCentroids,
            N, K, D, deviceAssignments, deviceAssignmentsChanged);
        tm->Stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());


        // Reset delta
        delta = 0;
        unsigned int* deviceDelta;
        CUDA_CHECK(cudaMalloc((void**)&deviceDelta, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy((void*)deviceDelta, (const void*)&delta, sizeof(unsigned int), cudaMemcpyHostToDevice));

        tm->Start();
        compute_delta << <blocksPerGridDelta, threadsPerBlockDelta, sharedMemSizeDelta >> > (deviceAssignmentsChanged, blocksPerGrid, deviceDelta);
        tm->Stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy((void*)&delta, (const void*)deviceDelta, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree((void*)deviceDelta));

        // Here we should recompute centroids based on new assignments
        CUDA_CHECK(cudaMemset((void*)deviceAssignmentsChanged, 0, sizeof(unsigned int) * blocksPerGrid));
        CUDA_CHECK(cudaMemcpy((void*)assignments, (const void*)deviceAssignments, assignmentsSize, cudaMemcpyDeviceToHost));

        int start_idx = iter == 0 ? K : 0;
        for (int i = start_idx; i < N; ++i)
        {
            int idx = assignments[i];

            for (int d = 0; d < D; ++d)
            {
                newClusters[idx * D + d] += datapoints[i * D + d];
            }
            clustersSizes[idx] += 1;
        }
        if (clustersSizes[0] == N)
        {
            printf("i=%d: All points assigned to cluster 0\n", iter);
        }

        for (int k = 0; k < K; ++k)
        {
            for (int d = 0; d < D; ++d)
            {
                if (clustersSizes[k] > 0) centroids[k * D + d] = newClusters[k * D + d] / clustersSizes[k];
				else centroids[k * D + d] = 0.0;
                newClusters[k * D + d] = 0.0;
            }
            clustersSizes[k] = 0;
        }

        printf("Iteration: %d, changes: %d\n", iter, delta);

    }

    if(D == 3) render(datapoints, deviceDatapoints, deviceAssignments, N, K);

    // CUDA_CHECK(cudaMemcpy((void*)assignments, (const void*)deviceAssignments, assignmentsSize, cudaMemcpyDeviceToHost));
    //CUDA_CHECK(cudaMemcpy((void*)centroids, (const void*)deviceCentroids, centroidsSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree((void*)deviceDatapoints));
    CUDA_CHECK(cudaFree((void*)deviceCentroids));
    CUDA_CHECK(cudaFree((void*)deviceAssignments));
    CUDA_CHECK(cudaFree((void*)deviceAssignmentsChanged));
    free(clustersSizes);
    free(newClusters);
}
