#ifndef XAYAH_EXAMPLES_SMOKE_WIND_TRAJECTORY_OPTIMIZATION_TASK_H
#define XAYAH_EXAMPLES_SMOKE_WIND_TRAJECTORY_OPTIMIZATION_TASK_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::smoke::examples::wind_trajectory_optimization::task_cuda {

    struct Grid {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        float cell_size;
    };

    struct VectorField {
        float* x;
        float* y;
        float* z;
    };

    struct ConstAdjointVectorField {
        const double* x;
        const double* y;
        const double* z;
    };

    void launch_write_control(cudaStream_t stream, Grid grid, std::uint32_t control_step, std::uint32_t trajectory_steps, const float* keyframes, float density_source_rate, float temperature_source_rate, float* density_source, float* temperature_source, VectorField external_acceleration);
    void launch_write_control_tangent(cudaStream_t stream, Grid grid, std::uint32_t control_step, std::uint32_t trajectory_steps, const float* keyframes, float* density_source, float* temperature_source, VectorField external_acceleration);
    void launch_density_energy(cudaStream_t stream, std::uint32_t cell_count, const float* density, double* energy);
    void launch_density_loss(cudaStream_t stream, std::uint32_t cell_count, double inverse_target_frame_energy, const float* density, const float* target_density, double* loss);
    void launch_density_loss_seed(cudaStream_t stream, std::uint32_t cell_count, double inverse_target_frame_energy, const float* density, const float* target_density, double* density_seed, double* loss);
    void launch_density_tangent_inner_product(cudaStream_t stream, std::uint32_t cell_count, const double* density_seed, const float* density_tangent, double* result);
    void launch_accumulate_keyframe_gradient(cudaStream_t stream, Grid grid, std::uint32_t control_step, std::uint32_t trajectory_steps, ConstAdjointVectorField acceleration_adjoint, double* keyframe_gradient);

} // namespace xayah::smoke::examples::wind_trajectory_optimization::task_cuda

#endif
