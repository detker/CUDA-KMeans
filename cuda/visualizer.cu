#include "visualizer.cuh"
#include "viz.cuh"
#include <cfloat>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdio>

void GPUVisualizer::visualize(const double* datapoints, const unsigned char* assignments,
                              int N, int K, int D)
{
    if (!canVisualize(D)) {
        printf("GPU Visualizer: Can only visualize 3D data.\n");
        return;
    }
    
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;
    float minz = FLT_MAX, maxz = -FLT_MAX;
    
    // Copy data to host to compute bounds
    double* host_data = new double[N * D];
    cudaMemcpy(host_data, datapoints, N * D * sizeof(double), cudaMemcpyDeviceToHost);
    
    for(int n = 0; n < N; ++n) {
        float x = (float)host_data[0 * N + n];
        float y = (float)host_data[1 * N + n];
        float z = (float)host_data[2 * N + n];
        
        minx = std::min(minx, x); maxx = std::max(maxx, x);
        miny = std::min(miny, y); maxy = std::max(maxy, y);
        minz = std::min(minz, z); maxz = std::max(maxz, z);
    }
    
    delete[] host_data;
    
    // printf("GPU Visualizer: Rendering %d points in %d clusters\n", N, K);
    render(datapoints, assignments, N, K, minx, maxx, miny, maxy, minz, maxz);
}

CPUVisualizer::~CPUVisualizer()
{
    freeGPUMem();
}

void CPUVisualizer::allocateGPUMem(int N, int D)
{
    freeGPUMem();
    
    cudaMalloc(&d_datapoints, N * D * sizeof(double));
    cudaMalloc(&d_assignments, N * sizeof(unsigned char));
    
    allocated_N = N;
    allocated_D = D;
}

void CPUVisualizer::freeGPUMem()
{
    if (d_datapoints) {
        cudaFree(d_datapoints);
        d_datapoints = nullptr;
    }
    if (d_assignments) {
        cudaFree(d_assignments);
        d_assignments = nullptr;
    }
    allocated_N = 0;
    allocated_D = 0;
}

void CPUVisualizer::visualize(const double* datapoints, const unsigned char* assignments,
                              int N, int K, int D)
{
    if (!canVisualize(D)) {
        printf("CPU Visualizer: Can only visualize 3D data.\n");
        return;
    }
    
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;
    float minz = FLT_MAX, maxz = -FLT_MAX;
    
    for(int n = 0; n < N; ++n) {
        float x = (float)datapoints[0 * N + n];
        float y = (float)datapoints[1 * N + n];
        float z = (float)datapoints[2 * N + n];
        
        minx = std::min(minx, x); maxx = std::max(maxx, x);
        miny = std::min(miny, y); maxy = std::max(maxy, y);
        minz = std::min(minz, z); maxz = std::max(maxz, z);
    }
    
    allocateGPUMem(N, D);
    
    cudaMemcpy(d_datapoints, datapoints, N * D * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assignments, assignments, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // printf("CPU Visualizer: Rendering %d points in %d clusters\n", N, K);
    render(d_datapoints, d_assignments, N, K, minx, maxx, miny, maxy, minz, maxz);
}

std::unique_ptr<IVisualizer> VisualizerFactory::create(Type type, int D)
{
    switch(type) {
        case Type::CPU_type:
            if (D == 3) {
                return std::make_unique<CPUVisualizer>();
            }
            break;
        case Type::GPU_type:
            if (D == 3) {
                return std::make_unique<GPUVisualizer>();
            }
            break;
    }
    return nullptr;
}