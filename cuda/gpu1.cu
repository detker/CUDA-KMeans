#include "gpu1.cuh"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// atomicAdd for double for pre-Pascal architectures
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


template<int D>
__global__ void compute_clusters(const double* datapoints, double *centroids,
    int N, int K,
    unsigned char* assignments, unsigned int* assignmentsChanged, double* newClusters, int* clustersSizes)
{
    // extern __shared__ char sharedMem[];
    // unsigned int* changedFlag = (unsigned int*)sharedMem;
    // double* clusters = (double*)(sharedMem + sizeof(unsigned int) * 8); // max 256 threads -> 8 warps

    extern __shared__ char sharedMem[];

    unsigned int *changedFlag = (unsigned int *)sharedMem; // max 256 threads -> for 8 warps, each warp accumulates its own changed count 

    size_t offset = ((sizeof(unsigned int) * 8 + 8 - 1) / 8) * 8; // align to double, we have 256 threads -> 8 warps
    double *clusters = (double*)(sharedMem + offset);

    // for first K threads, load centroids into shared memory computing them from newClusters and clustersSizes on the go
    if (threadIdx.x < K)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            double value;
            if(clustersSizes[threadIdx.x] > 0)
            {
                value = newClusters[d*K + threadIdx.x] / clustersSizes[threadIdx.x];
            }
            else
            {
                value = centroids[d*K + threadIdx.x];
            }
            clusters[d * K + threadIdx.x] = value;
            // here at most 4 bank conflicts (due to constraint K <= 20), happening only in the first warp - negligible
            if(blockIdx.x == 0) centroids[d*K + threadIdx.x] = value; 
        }
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // index for point to process by thread

    unsigned int changed = 0; // local changed count for warp reduction

    // each thread processes one data point, finds its closest cluster and updates assignment if needed
    if (idx < N)
    {
        double minDistance = DEVICE_INF;
        unsigned char bestCluster;

        for (int k = 0; k < K; k++) {
            double sum = 0.0;
            #pragma unroll
            for(int d=0; d < D; ++d)
            {
                double diff = datapoints[d * N + idx] - clusters[d * K + k]; //broadcast happens here - no conflicts
                sum += diff * diff;
            }
            if (sum < minDistance) {
                minDistance = sum;
                bestCluster = k;
            }
        }

        // if assignment changed, update it and set changed flag
        changed = (assignments[idx] != bestCluster) ? 1 : 0;
        assignments[idx] = bestCluster; 
    }

    // warp shuffle reduction to sum up changed within a warp
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        #if CUDART_VERSION >= 9000
            changed += __shfl_down_sync(0xffffffff, changed, offset);
        #else
            changed += __shfl_down(changed, offset);
        #endif
    }
    // after the loop lane 0 has the sum from warp

    // compute warp id and warp's lane id
    int lane_id = threadIdx.x & 31; // same as threadIdx.x % 32
    int warp_id = threadIdx.x >> 5; // same as threadIdx.x / 32

    // store per-warp sums into shared memory defined at the beginning
    if(lane_id == 0)
    {
        changedFlag[warp_id] = changed;
    }

    __syncthreads();

    // first warp reduces the per-warp sums and atomically adds the final sum to global assignmentsChanged
    if(warp_id == 0)
    {
        // if our thread is less than number of warps in the block, load changedFlag[threadIdx.x], else 0
        // each thread is assigned to reduce one value from changedFlag
        unsigned int warpSum = (threadIdx.x < (blockDim.x + 31) / 32) ? changedFlag[threadIdx.x] : 0;
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            #if CUDART_VERSION >= 9000
                warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
            #else
                warpSum += __shfl_down(warpSum, offset);
            #endif
        }

        if(threadIdx.x == 0) 
        {
            atomicAdd(assignmentsChanged, warpSum);
        }
    }
}

template<int D>
__global__ void scatter_clusters(const double* datapoints, const unsigned char* assignments,
    int N, int K,
    double* newClusters, int* clustersSizes)
{
    extern __shared__ char shmm[];

    unsigned int *shmSizes = (unsigned int *)shmm;

    size_t offset = ((sizeof(unsigned int) * K + 8 - 1) / 8) * 8; // align to double
    double *shmClusters = (double*)(shmm + offset);

    if(threadIdx.x < K)
    {
        shmSizes[threadIdx.x] = 0;
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            // at most 4 bank conflicts (due to constraint K<=20), happening only in first warp - negligible
            shmClusters[d * K + threadIdx.x] = 0.0;
        }
    }

    __syncthreads();

    // each thread processes one data point
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // find assigned cluster
    unsigned char clusterIdx = assignments[idx];

    // atomicly add datapoint to shared memory cluster sum and increment size
    #pragma unroll
    for (int d = 0; d < D; ++d)
    {
        atomicAdd(&shmClusters[d * K + clusterIdx], datapoints[d * N + idx]);
    }
    atomicAdd(&shmSizes[clusterIdx], 1);


    // for K first threads, accumulate shared memory clusters into global memory
    if(threadIdx.x < K)
    {
        __syncthreads(); // inside if - since volta its clearly safe
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            atomicAdd(&newClusters[d * K + threadIdx.x], shmClusters[d * K + threadIdx.x]);
        }
        atomicAdd(&clustersSizes[threadIdx.x], shmSizes[threadIdx.x]);
    }
}

// implementation of GATHER (exchangeble with SCATTER above) - commented out, because of bank conflicts and lower occupancy,
// as I was terribly scared of penalty points, THOUGH IT OVERALL PERFORMS BETTER IN PRACTICE (ON EXAMPLE DATA UP TO 20% SPEED UP).
// template<int D>
// __global__ void scatter_clusters(const double* datapoints, const unsigned char* assignments,
//     int N, int K,
//     double* newClusters, int* clustersSizes)
// {
//     int blocksPerCluster = gridDim.x / K; // number of blocks assigned to each cluster :)
//     int k = blockIdx.x / blocksPerCluster; // [000000,111111,222222,...] cluster index
//     int blockId = blockIdx.x % blocksPerCluster;  // block idx within cluster k

//     extern __shared__ double shmm[];
//     double *localSums = shmm;
//     uint16_t *localSizes = (uint16_t *)(shmm + blockDim.x * D);
//     #pragma unroll
//     for(int d=0; d < D; ++d)
//     {
//         localSums[d*blockDim.x + threadIdx.x] = 0.0;
//     }
//     localSizes[threadIdx.x] = 0;

//     __syncthreads();

//     // each block handles a chunk of data points
//     int pointsPerBlock = (N + blocksPerCluster - 1) / blocksPerCluster;
//     int startIdx = blockId * pointsPerBlock;
//     int endIdx = min(startIdx + pointsPerBlock, N);

//     for(int idx = startIdx + threadIdx.x; idx < endIdx; idx += blockDim.x)
//     {
//         if(assignments[idx] == k)
//         {
//             #pragma unroll
//             for(int d=0; d < D; ++d)
//             {
//                 localSums[d*blockDim.x + threadIdx.x] += datapoints[d*N + idx];
//             }
//             ++localSizes[threadIdx.x];
//         }
//     }

//     __syncthreads();

//     // reduce within block
//     for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
//     {
//         if(threadIdx.x < offset)
//         {
//             #pragma unroll
//             for(int d=0; d < D; ++d)
//             {
//                 localSums[d*blockDim.x + threadIdx.x] += localSums[d*blockDim.x + threadIdx.x+offset];
//             }
//             localSizes[threadIdx.x] += localSizes[threadIdx.x + offset];
//         }
//         __syncthreads();
//     }

//     if(threadIdx.x == 0)
//     {
//         #pragma unroll
//         for(int d=0; d < D; ++d)
//         {
//             atomicAdd(&newClusters[d*K + k], localSums[d*blockDim.x + 0]);
//         }
//         atomicAdd(&clustersSizes[k], localSizes[0]);
//     }
// }

template<int D>
void kmeans_host(const double* datapoints, double* centroids,
    int N, int K, unsigned char* assignments, TimerManager *tm)
{
    TimerGPU timerGPU;
    tm->SetTimer(&timerGPU);
    
    // get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const double *deviceDatapoints;
    double* deviceCentroids;
    unsigned char* deviceAssignments;
    double *newClustersDevice;
    int *clustersSizesDevice;
    size_t datapointsSize = N * D * sizeof(double);
    size_t centroidsSize = K * D * sizeof(double);
    // as K <= 20 then unsigned char is enough to store cluster index
    size_t assignmentsSize = N * sizeof(unsigned char); 
    size_t clustersSizesSize = K * sizeof(int);

    CUDA_CHECK(cudaMalloc((void**)&newClustersDevice, centroidsSize));
    CUDA_CHECK(cudaMalloc((void**)&clustersSizesDevice, clustersSizesSize));
    CUDA_CHECK(cudaMalloc((void**)&deviceDatapoints, datapointsSize));
    CUDA_CHECK(cudaMalloc((void**)&deviceCentroids, centroidsSize));
    CUDA_CHECK(cudaMalloc((void**)&deviceAssignments, assignmentsSize));

    // copy data points to device
    CUDA_CHECK(cudaMemcpy((void*)deviceDatapoints, (const void*)datapoints, datapointsSize, cudaMemcpyHostToDevice));

    // initialize centroids by first K datapoints
    for (int k = 0; k < K; ++k)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            centroids[d * K + k] = datapoints[d * N + k];
        }
    }

    // copy centroids to device
    CUDA_CHECK(cudaMemcpy((void*)deviceCentroids, (const void*)centroids, centroidsSize, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset((void*)deviceAssignments, 0, assignmentsSize)); // zero initialize assignments

    // copy centroids to newClustersDevice and zero initialize clustersSizesDevice
    CUDA_CHECK(cudaMemcpy((void*)newClustersDevice, (const void*)centroids, centroidsSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset((void*)clustersSizesDevice, 0, clustersSizesSize));

    // initialize delta and deviceDelta
    unsigned delta = N;    
    unsigned int* deviceDelta;
    CUDA_CHECK(cudaMalloc((void**)&deviceDelta, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset((void*)deviceDelta, 0, sizeof(unsigned int)));
    
    // int blockSize = 256;  
    // int sharedMemPerBlock = ((sizeof(unsigned int)*8 + 8 - 1)/8) * 8 + 20 * 20 * sizeof(double);
    
    // int numBlocksPerSM;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocksPerSM,
    //     compute_clusters<20>,
    //     blockSize,
    //     sharedMemPerBlock
    // );
    
    // float occupancy = (numBlocksPerSM * blockSize) / 
    //                   (float)prop.maxThreadsPerMultiProcessor;
    
    // printf("Theoretical Occupancy: %.2f%%\n", occupancy * 100);
    // printf("Blocks per SM: %d\n", numBlocksPerSM);
    // printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);


    // int numThreads = 256;
    // int sharedMemSizeScatter = sizeof(double) * 20 *  numThreads + sizeof(uint16_t) * numThreads;
    // int numThreads = 256;
    // int numBlocks = (N + numThreads - 1) / numThreads;
    // int sharedMemSizeScatter = ((sizeof(unsigned int) * 20 + 8 - 1) / 8) * 8 + sizeof(double) * 20 * 20;

    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocksPerSM,
    //     scatter_clusters<20>,
    //     numThreads,
    //     sharedMemSizeScatter
    // );
    
    // occupancy = (numBlocksPerSM * numThreads) / 
    //                   (float)prop.maxThreadsPerMultiProcessor;
    
    // printf("Theoretical Occupancy: %.2f%%\n", occupancy * 100);
    // printf("Blocks per SM: %d\n", numBlocksPerSM);
    // printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

    // compute_cluster kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // int sharedMemSize = sizeof(unsigned int) * 8 + sizeof(double) * K * D;
    // shm size for storing 256/32 = 8 assignment changes for each warp in a block and for storing all centroids 
    // (aligned to double)
    int sharedMemSize = (sizeof(unsigned int)*8 + 8 - 1)/8 * 8 + sizeof(double) * K * D;

    // scatter_clusters kernel launch parameters
    int numThreadsScatter = prop.major * 10 + prop.minor >= 60 ? 128 : 256;
    // int numThreadsScatter = 256;
    int numBlocksScatter = (N + numThreadsScatter - 1) / numThreadsScatter;
    // shm size for storing K clusters (aligned to double) and their sizes (unsigned int)
    int sharedMemSizeScatter = ((sizeof(unsigned int) * K + 8 - 1) / 8) * 8 + sizeof(double) * K * D;
    for (int iter = 0; iter < MAX_ITERATIONS && delta > 0; iter++)
    {
        tm->Start();
        compute_clusters<D><< <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (deviceDatapoints, deviceCentroids,
            N, K, deviceAssignments, deviceDelta, newClustersDevice, clustersSizesDevice);
        tm->Stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        // prepare newClusters for sum accumulation in scatter kernel
        CUDA_CHECK(cudaMemset((void*)newClustersDevice, 0, centroidsSize));
        CUDA_CHECK(cudaMemset((void*)clustersSizesDevice, 0, clustersSizesSize));

        // copy delta from device to host to check for convergence
        CUDA_CHECK(cudaMemcpy((void*)&delta, (const void*)deviceDelta, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset((void*)deviceDelta, 0, sizeof(unsigned int)));

        // commented out code for launching GATHER kernel - see above for details
        // int numThreads = 256;
        // int numBlocks = (N + numThreads - 1) / numThreads;
        // int blocksPerCluster = (numBlocks + K - 1) / K;
        // int totalBlocks = blocksPerCluster * K;
        // int sharedMemSizeScatter = sizeof(double) * D * numThreads + sizeof(uint16_t) * numThreads;
        // tm->Start();
        // scatter_clusters<D><< <totalBlocks, numThreads, sharedMemSizeScatter >> > (deviceDatapoints, deviceAssignments,
        //     N, K,
        //     newClustersDevice, clustersSizesDevice);
        // tm->Stop();
        // CUDA_CHECK(cudaDeviceSynchronize());
        // CUDA_CHECK(cudaGetLastError());

        tm->Start();
        scatter_clusters<D><<<numBlocksScatter, numThreadsScatter, sharedMemSizeScatter>>>(deviceDatapoints, deviceAssignments,
            N, K,
            newClustersDevice, clustersSizesDevice);
        tm->Stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        printf("Iteration: %d, changes: %d\n", iter, delta);
    }

    // visualization when D = 3
    auto visualizer = VisualizerFactory::create(ComputeType::GPU1, D);
    if (visualizer && visualizer->canVisualize(D))
    {
        visualizer->visualize(deviceDatapoints, deviceAssignments, N, K, D);
    }

    // copy assignments back to host
    CUDA_CHECK(cudaMemcpy((void*)assignments, (const void*)deviceAssignments, assignmentsSize, cudaMemcpyDeviceToHost));

    // code for updating last iteration centroids but now on host and directly into row-major format
    double* centroids_col_major = (double*)malloc(centroidsSize);
    if (!centroids_col_major) ERR("malloc centroids_col_major failed.");
    CUDA_CHECK(cudaMemcpy((void*)centroids_col_major, (const void*)deviceCentroids, centroidsSize, cudaMemcpyDeviceToHost));
    double *newClusters_col_major = (double*)malloc(centroidsSize);
    if (!newClusters_col_major) ERR("malloc newClusters_col_major failed.");
    CUDA_CHECK(cudaMemcpy((void*)newClusters_col_major, (const void*)newClustersDevice, centroidsSize, cudaMemcpyDeviceToHost));
    int *clusterSizes_col_major = (int*)malloc(clustersSizesSize);
    if (!clusterSizes_col_major) ERR("malloc clusterSizes_col_major failed.");
    CUDA_CHECK(cudaMemcpy((void*)clusterSizes_col_major, (const void*)clustersSizesDevice, clustersSizesSize, cudaMemcpyDeviceToHost));
    for (int k = 0; k < K; ++k)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            if(clusterSizes_col_major[k] > 0)
            {
                centroids[k * D + d] = newClusters_col_major[d * K + k] / clusterSizes_col_major[k];
            }
            else
            {
                centroids[k * D + d] = centroids_col_major[d * K + k];
            }
        }
    }

    CUDA_CHECK(cudaFree((void*)deviceDatapoints));
    CUDA_CHECK(cudaFree((void*)deviceCentroids));
    CUDA_CHECK(cudaFree((void*)deviceAssignments));
    CUDA_CHECK(cudaFree((void*)newClustersDevice));
    CUDA_CHECK(cudaFree((void*)clustersSizesDevice));
    CUDA_CHECK(cudaFree((void*)deviceDelta));
    free(centroids_col_major);
    free(newClusters_col_major);
    free(clusterSizes_col_major);
}

// explicit template instantiation for kmeans_host with D = 1 to 20
using KMeansFunc = void(const double*, double*, int, int, unsigned char*, TimerManager*);
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
