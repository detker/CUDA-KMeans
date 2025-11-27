#pragma once
#include <memory>
#include "enum_types.h"

class IVisualizer
{
    public:
        virtual void visualize(const double* points, const unsigned char* assignments, int N, int K, int D) = 0;
        virtual ~IVisualizer() = default;
        virtual bool canVisualize(int D) const = 0;
};

class VisualizerFactory 
{
    public:
        ComputeType compute_method;

    static std::unique_ptr<IVisualizer> create(ComputeType type, int D);
};

class GPUVisualizer : public IVisualizer
{
    public:
        GPUVisualizer() = default;
        ~GPUVisualizer() override = default;

        void visualize(const double* points, const unsigned char* assignments, int N, int K, int D) override;
        bool canVisualize(int D) const override {return D == 3; }
};

class CPUVisualizer : public IVisualizer
{
    public:
        CPUVisualizer() = default;
        ~CPUVisualizer() override;

        void visualize(const double* points, const unsigned char* assignments, int N, int K, int D) override;
        bool canVisualize(int D) const override {return D == 3; }

    private:
        double *d_datapoints = nullptr;
        unsigned char *d_assignments = nullptr;
        size_t allocated_N = 0;
        size_t allocated_D = 0;

        void allocateGPUMem(int N, int D);
        void freeGPUMem();
};