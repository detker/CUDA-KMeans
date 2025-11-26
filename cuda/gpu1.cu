#include "gpu1.cuh"

template<int D>
__global__ void update_centroids(double* centroids, const double* newClusters, 
                                  const int* clustersSizes, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = K * D;
    
    if (idx >= total_elements) return;
    
    int k = idx % K;        // cluster index
    int d = idx / K;        // dimension index
    
    if (clustersSizes[k] > 0)
    {
        centroids[d * K + k] = newClusters[d * K + k] / clustersSizes[k];
    }
}

template<int D>
__global__ void compute_clusters(const double* datapoints, double *centroids,
    int N, int K,
    int* assignments, unsigned int* assignmentsChanged, double* newClusters, int* clustersSizes)
{
    extern __shared__ char sharedMem[];

    unsigned int* changedFlag = (unsigned int*)sharedMem;

    double* clusters = (double*)(sharedMem + sizeof(unsigned int) * 32);

    if (threadIdx.x < K)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            clusters[d * K + threadIdx.x] = centroids[d*K + threadIdx.x];
        }
    }
    // changedFlag[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int changed = 0;

    if (idx < N)
    {
        double minDistance = DEVICE_INF;
        int bestCluster = 0;

        for (int k = 0; k < K; k++) {
            // double distance;
            // // compute_distance_l2(&datapoints[idx * D], &clusters[k * D], D, &distance);
            // compute_distance_l2(datapoints, clusters, idx, k, N, K, D, &distance);
            double sum = 0.0;
            #pragma unroll
            for(int d=0; d < D; ++d)
            {
                double diff = datapoints[d * N + idx] - clusters[d * K + k];
                sum += diff * diff;
            }
            // if (distance < minDistance) {
            if (sum < minDistance) {
                minDistance = sum;
                bestCluster = k;
            }
        }

        // if (assignments[idx] != bestCluster)
        // {
        //     assignments[idx] = bestCluster;
        //     changedFlag[threadIdx.x] = 1;
        // }
        changed = (assignments[idx] != bestCluster) ? 1 : 0;
        assignments[idx] = bestCluster; 
    }

    // warp shuffle
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        changed += __shfl_down_sync(0xffffffff, changed, offset);
    }
    // lane 0 has the sum from warp

    int lane_id = threadIdx.x & 31; // same as threadIdx.x % 32
    int warp_id = threadIdx.x >> 5; // same as threadIdx.x / 32

    if(lane_id == 0)
    {
        changedFlag[warp_id] = changed;
    }

    __syncthreads();

    if(warp_id == 0)
    {
        unsigned int warpSum = (threadIdx.x < (blockDim.x + 31) / 32) ? changedFlag[threadIdx.x] : 0; // jesli mamy thread mniejszy niz liczba warpow w bloku to bierzemy wartosc z changedFlag
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }

        if(threadIdx.x == 0) // equivalent to if(threadIdx.x == 0)
        {
            // assignmentsChanged[blockIdx.x] = warpSum;
            atomicAdd(assignmentsChanged, warpSum);
        }
    }
}

template<int D>
__global__ void scatter_clusters(const double* datapoints, const int* assignments,
    int N, int K,
    double* newClusters, int* clustersSizes, double* centroids)
{
    int blocksPerCluster = gridDim.x / K; // number of blocks assigned to each cluster :)
    int k = blockIdx.x / blocksPerCluster; // [000000,111111,222222,...] cluster index
    int blockId = blockIdx.x % blocksPerCluster;  // block idx within cluster k

    extern __shared__ double shmm[];
    double *localSums = shmm;
    unsigned int *localSizes = (unsigned int *)(shmm + blockDim.x * D);
    #pragma unroll
    for(int d=0; d < D; ++d)
    {
        // localSums[D*threadIdx.x + d] = 0.0;
        localSums[d*blockDim.x + threadIdx.x] = 0.0;
    }
    localSizes[threadIdx.x] = 0;

    __syncthreads();

    // each block handles a chunk of data points
    int pointsPerBlock = (N + blocksPerCluster - 1) / blocksPerCluster;
    int startIdx = blockId * pointsPerBlock;
    int endIdx = min(startIdx + pointsPerBlock, N);

    for(int idx = startIdx + threadIdx.x; idx < endIdx; idx += blockDim.x)
    {
        if(assignments[idx] == k)
        {
            #pragma unroll
            for(int d=0; d < D; ++d)
            {
                // localSums[D*threadIdx.x + d] += datapoints[D*idx + d];
                // localSums[D*threadIdx.x + d] += datapoints[d*N + idx];
                localSums[d*blockDim.x + threadIdx.x] += datapoints[d*N + idx];
            }
            ++localSizes[threadIdx.x];
        }
    }

    __syncthreads();

    // reduce within block
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            #pragma unroll
            for(int d=0; d < D; ++d)
            {
                // localSums[D*threadIdx.x + d] += localSums[D*(threadIdx.x + offset) + d];
                localSums[d*blockDim.x + threadIdx.x] += localSums[d*blockDim.x + threadIdx.x+offset];
            }
            localSizes[threadIdx.x] += localSizes[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        #pragma unroll
        for(int d=0; d < D; ++d)
        {
            // atomicAdd(&newClusters[D*k + d], localSums[d]);
            atomicAdd(&newClusters[d*K + k], localSums[d*blockDim.x + 0]);
        }
        atomicAdd(&clustersSizes[k], localSizes[0]);
    }
}

template<int D>
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, int* assignments, TimerManager *tm)
{
    TimerGPU timerGPU;
    tm->SetTimer(&timerGPU);

    const double *deviceDatapointsRowMajor;
    const double *deviceDatapoints;
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

    double *newClustersDevice;
    int *clustersSizesDevice;
    CUDA_CHECK(cudaMalloc((void**)&newClustersDevice, centroidsSize));
    CUDA_CHECK(cudaMalloc((void**)&clustersSizesDevice, clustersSizesSize));

    CUDA_CHECK(cudaMalloc((void**)&deviceDatapoints, datapointsSize));

    CUDA_CHECK(cudaMalloc((void**)&deviceCentroids, centroidsSize));
    CUDA_CHECK(cudaMalloc((void**)&deviceAssignments, assignmentsSize));

    CUDA_CHECK(cudaMemcpy((void*)deviceDatapoints, (const void*)datapoints, datapointsSize, cudaMemcpyHostToDevice));

    // AoS -> SoA
    for (int k = 0; k < K; ++k)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            // centroids[d * K + k] = datapoints[k * D + d];
            centroids[d * K + k] = datapoints[d * N + k];
        }
    }

    CUDA_CHECK(cudaMemcpy((void*)deviceCentroids, (const void*)centroids, centroidsSize, cudaMemcpyHostToDevice));

    // Initialize assignments
    CUDA_CHECK(cudaMemset((void*)deviceAssignments, 0, assignmentsSize));


    CUDA_CHECK(cudaMemcpy((void*)newClustersDevice, (const void*)centroids, centroidsSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset((void*)clustersSizesDevice, 0, clustersSizesSize));

    unsigned delta = N;

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = sizeof(unsigned int) * 32 + sizeof(double) * K * D;
    
    int threadsPerBlockDelta = 512;
    int blocksPerGridDelta = 1;
    int sharedMemSizeDelta = sizeof(unsigned int) * threadsPerBlockDelta;

    CUDA_CHECK(cudaMalloc((void**)&deviceAssignmentsChanged, sizeof(unsigned int) * blocksPerGrid));
    CUDA_CHECK(cudaMemset((void*)deviceAssignmentsChanged, 0, sizeof(unsigned int) * blocksPerGrid));

    unsigned int* deviceDelta;
    CUDA_CHECK(cudaMalloc((void**)&deviceDelta, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset((void*)deviceDelta, 0, sizeof(unsigned int)));

    for (int iter = 0; iter < MAX_ITERATIONS && delta > 0; iter++)
    {
        tm->Start();
        compute_clusters<D><< <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (deviceDatapoints, deviceCentroids,
            N, K, deviceAssignments, deviceDelta, newClustersDevice, clustersSizesDevice);
        tm->Stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset((void*)newClustersDevice, 0, centroidsSize));
        CUDA_CHECK(cudaMemset((void*)clustersSizesDevice, 0, clustersSizesSize));

        CUDA_CHECK(cudaMemcpy((void*)&delta, (const void*)deviceDelta, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset((void*)deviceDelta, 0, sizeof(unsigned int)));

        int numThreads = 256;
        
        int numBlocks = (N + numThreads - 1) / numThreads;
        int blocksPerCluster = (numBlocks + K - 1) / K;
        int totalBlocks = blocksPerCluster * K;
        int sharedMemSizeScatter = sizeof(double) * D * numThreads + sizeof(unsigned int) * numThreads;
        tm->Start();
        scatter_clusters<D><< <totalBlocks, numThreads, sharedMemSizeScatter >> > (deviceDatapoints, deviceAssignments,
            N, K,
            newClustersDevice, clustersSizesDevice, deviceCentroids);
        tm->Stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        // Update centroids
        int centroidThreads = 256;
        int centroidBlocks = (K*D + centroidThreads - 1) / centroidThreads;
        
        tm->Start();
        update_centroids<D><<<centroidBlocks, centroidThreads>>>(
            deviceCentroids, newClustersDevice, clustersSizesDevice, K);
        tm->Stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        printf("Iteration: %d, changes: %d\n", iter, delta);

    }

    if (D == 3) 
    {
        float minx, maxx, miny, maxy, minz, maxz;
        compute_bounds(datapoints, N, minx, maxx, miny, maxy, minz, maxz);
        render(deviceDatapoints, deviceAssignments, N, K, minx, maxx, miny, maxy, minz, maxz);
    }

    CUDA_CHECK(cudaMemcpy((void*)assignments, (const void*)deviceAssignments, assignmentsSize, cudaMemcpyDeviceToHost));

    double* centroids_col_major = (double*)malloc(centroidsSize);
    if (!centroids_col_major) ERR("malloc centroids_col_major failed.");
    CUDA_CHECK(cudaMemcpy((void*)centroids_col_major, (const void*)deviceCentroids, centroidsSize, cudaMemcpyDeviceToHost));

    // Convert from column-major (SoA) to row-major (AoS) for output
    for (int k = 0; k < K; ++k)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            centroids[k * D + d] = centroids_col_major[d * K + k];
        }
    }

    CUDA_CHECK(cudaFree((void*)deviceDatapoints));
    CUDA_CHECK(cudaFree((void*)deviceCentroids));
    CUDA_CHECK(cudaFree((void*)deviceAssignments));
    CUDA_CHECK(cudaFree((void*)deviceAssignmentsChanged));
    CUDA_CHECK(cudaFree((void*)newClustersDevice));
    CUDA_CHECK(cudaFree((void*)clustersSizesDevice));
    CUDA_CHECK(cudaFree((void*)deviceDelta));
    free(centroids_col_major);
    free(clustersSizes);
    free(newClusters);
}

using KMeansFunc = void(const double*, double*, int, int, int*, TimerManager*);

template KMeansFunc kmeans_host<1>;
template KMeansFunc kmeans_host<2>;
template KMeansFunc kmeans_host<3>;
template KMeansFunc kmeans_host<4>;
template KMeansFunc kmeans_host<5>;
template KMeansFunc kmeans_host<6>;
template KMeansFunc kmeans_host<7>;
template KMeansFunc kmeans_host<8>;
template KMeansFunc kmeans_host<9>;
template KMeansFunc kmeans_host<10>;
template KMeansFunc kmeans_host<11>;
template KMeansFunc kmeans_host<12>;
template KMeansFunc kmeans_host<13>;
template KMeansFunc kmeans_host<14>;
template KMeansFunc kmeans_host<15>;
template KMeansFunc kmeans_host<16>;
template KMeansFunc kmeans_host<17>;
template KMeansFunc kmeans_host<18>;
template KMeansFunc kmeans_host<19>;
template KMeansFunc kmeans_host<20>;
