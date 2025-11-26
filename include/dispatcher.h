#include <stdio.h>
#include "gpu1.cuh"
#include "gpu2.cuh"
#include "cpu.h"
#include "timer.h"
#include "utils.h"

// compile-time sequence for dimension iteration
template<int... Dims>
struct DimSeq {};

template<int N, int... Dims>
struct MakeDimSeq : MakeDimSeq<N - 1, N - 1, Dims...> {};

template<int... Dims>
struct MakeDimSeq<0, Dims...> {
    using type = DimSeq<Dims...>;
};

template<int D>
struct KMeansDispatcher 
{
    static void dispatch(const double *datapoints, double *centroids, int *assignments,
                         int N, int K, unsigned char compute_method,
                         TimerManager *tm) 
    {
        if (compute_method == GPU1)
        {
            printf("Running K-means on gpu1 (parallel, custom kernels)...\n");
            kmeans_host<D>(datapoints, centroids, N, K, assignments, tm);
        }
        else if (compute_method == GPU2)
        {
            printf("Running K-means on gpu2 (parallel, thrust)...\n");
            thrust_kmeans_host<D>(datapoints, centroids, N, K, assignments, tm);
        }
        else // CPU
        {
            printf("Running K-means on CPU (sequential)...\n");
            seq_kmeans<D>(datapoints, centroids, N, K, assignments, tm);
        }
    }
};

// Runtime dimension dispatcher using template recursion
template<int CurrentD, int MaxD>
struct RuntimeDispatcher
{
    template<typename Func, typename... Args>
    static void dispatch(int D, Func&& func, Args&&... args)
    {
        if (D == CurrentD) 
        {
            func.template operator()<CurrentD>(std::forward<Args>(args)...);
        }
        else 
        {
            RuntimeDispatcher<CurrentD + 1, MaxD>::dispatch(D, std::forward<Func>(func), std::forward<Args>(args)...);
        }
    }
};

template<int MaxD>
struct RuntimeDispatcher<MaxD, MaxD>
{
    template<typename Func, typename... Args>
    static void dispatch(int D, Func&& func, Args&&... args)
    {
        if (D == MaxD) 
        {
            func.template operator()<MaxD>(std::forward<Args>(args)...);
        } 
        // if D is out of bounds, handle error
        else 
        {
            fprintf(stderr, "Error: Dimension %d out of bounds (1-%d)\n", D, MaxD);
            exit(EXIT_FAILURE);
        }
    }
};

struct KMeansLauncher 
{
    const double *datapoints;
    double *centroids;
    int *assignments;
    int N; // number of datapoints
    int K; // number of centroids
    unsigned char compute_method;
    TimerManager *tm;

    template <int D>
    void operator()() const 
    {
        KMeansDispatcher<D>::dispatch(datapoints, centroids, assignments, N, K, compute_method, tm);
    }
};

void RunKMeans(const double *datapoints, double *centroids, int *assignments,
               int N, int K, int D, unsigned char compute_method,
               TimerManager *tm) 
{
    static constexpr int MAX_DIM = 20;

    KMeansLauncher launcher{datapoints, centroids, assignments, N, K, compute_method, tm};

    RuntimeDispatcher<1, MAX_DIM>::dispatch(D, launcher);
}
