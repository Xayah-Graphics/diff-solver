#ifndef XAYAH_EXAMPLES_CLOTH_WIND_TRAJECTORY_OPTIMIZATION_TASK_H
#define XAYAH_EXAMPLES_CLOTH_WIND_TRAJECTORY_OPTIMIZATION_TASK_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::cloth::examples::wind_trajectory_optimization::task_cuda {

    struct ConstField {
        const float* x;
        const float* y;
        const float* z;
    };

    struct Field {
        float* x;
        float* y;
        float* z;
    };

    void launch_write_control(cudaStream_t stream, std::uint32_t particle_count, std::uint32_t control_step, std::uint32_t trajectory_steps, const std::uint32_t* anchor_mask, const float* keyframes, Field external_forces);
    void launch_position_loss(cudaStream_t stream, std::uint32_t particle_count, double normalization, ConstField positions, ConstField target_positions, double* loss);
    void launch_position_loss_seed(cudaStream_t stream, std::uint32_t particle_count, float normalization, ConstField positions, ConstField target_positions, Field position_seed, double* loss);
    void launch_position_tangent_inner_product(cudaStream_t stream, std::uint32_t particle_count, ConstField position_seed, ConstField position_tangent, double* result);
    void launch_accumulate_keyframe_gradient(cudaStream_t stream, std::uint32_t particle_count, std::uint32_t control_step, std::uint32_t trajectory_steps, const std::uint32_t* anchor_mask, ConstField control_adjoint, double* keyframe_gradient);

} // namespace xayah::cloth::examples::wind_trajectory_optimization::task_cuda

#endif
