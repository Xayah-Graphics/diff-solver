#ifndef XAYAH_EXAMPLES_CLOTH_STRETCH_STIFFNESS_INVERSE_TASK_H
#define XAYAH_EXAMPLES_CLOTH_STRETCH_STIFFNESS_INVERSE_TASK_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::cloth::examples::stretch_stiffness_inverse::inverse_cuda {

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

    void launch_fill(cudaStream_t stream, std::uint32_t count, float value, float* output);
    void launch_position_loss(cudaStream_t stream, std::uint32_t particle_count, double normalization, ConstField positions, ConstField target_positions, double* loss);
    void launch_position_loss_seed(cudaStream_t stream, std::uint32_t particle_count, float normalization, ConstField positions, ConstField target_positions, Field position_seed, double* loss);
    void launch_position_tangent_inner_product(cudaStream_t stream, std::uint32_t particle_count, ConstField position_seed, ConstField position_tangent, double* result);
    void launch_sum(cudaStream_t stream, std::uint32_t count, const float* values, double* result);

} // namespace xayah::cloth::examples::stretch_stiffness_inverse::inverse_cuda

#endif
