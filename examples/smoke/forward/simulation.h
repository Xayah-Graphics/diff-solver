#ifndef XAYAH_EXAMPLES_SMOKE_FORWARD_SIMULATION_H
#define XAYAH_EXAMPLES_SMOKE_FORWARD_SIMULATION_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::smoke::examples::forward::simulation_cuda {

    struct Grid {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        float cell_size;
        float time_step;
    };

    struct Vector {
        float x;
        float y;
        float z;
    };

    struct VectorField {
        float* x;
        float* y;
        float* z;
    };

    struct ConstStaggeredVectorField {
        const float* x;
        const float* y;
        const float* z;
    };

    void launch_write_control(cudaStream_t stream, Grid grid, std::uint64_t step, float pulse_period, Vector left_center, Vector right_center, float source_radius, float density_source_rate, float temperature_source_rate, Vector left_acceleration, Vector right_acceleration, float* density_source, float* temperature_source, VectorField external_acceleration);
    void launch_reduce_metrics(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, const float* density, const float* temperature, ConstStaggeredVectorField pre_projection_velocity, ConstStaggeredVectorField post_projection_velocity, double* metrics);

} // namespace xayah::smoke::examples::forward::simulation_cuda

#endif
