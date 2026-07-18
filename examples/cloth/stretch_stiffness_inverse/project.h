#ifndef XAYAH_EXAMPLES_CLOTH_STRETCH_STIFFNESS_INVERSE_PROJECT_H
#define XAYAH_EXAMPLES_CLOTH_STRETCH_STIFFNESS_INVERSE_PROJECT_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::cloth::examples::stretch_stiffness_inverse::visualization_cuda {

    enum class SegmentStyle : std::uint32_t {
        estimate,
        target,
        bending,
    };

    void launch_segments(
        cudaStream_t stream,
        std::uint32_t spring_count,
        const float* position_x,
        const float* position_y,
        const float* position_z,
        const std::uint32_t* spring_first,
        const std::uint32_t* spring_second,
        const float* rest_lengths,
        float width,
        float strain_range,
        SegmentStyle style,
        void* output);

} // namespace xayah::cloth::examples::stretch_stiffness_inverse::visualization_cuda

#endif
