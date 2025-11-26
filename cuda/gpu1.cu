// gpu1.cu - CUDA implementation file
// This file is compiled by nvcc and provides explicit template instantiations

#include "gpu1.cuh"

// Explicit template instantiations for dimensions 1-20
template void kmeans_host<1>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<2>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<3>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<4>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<5>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<6>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<7>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<8>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<9>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<10>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<11>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<12>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<13>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<14>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<15>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<16>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<17>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<18>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<19>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
template void kmeans_host<20>(const double* datapoints, double* centroids, int N, int K, int* assignments, TimerManager *tm);
