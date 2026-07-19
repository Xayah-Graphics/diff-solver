#ifndef XAYAH_EXAMPLES_CLOTH_FORWARD_PROJECT_H
#define XAYAH_EXAMPLES_CLOTH_FORWARD_PROJECT_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::cloth::examples::forward::project_cuda {

    enum class SegmentStyle : std::uint32_t {
        stretch,
        bending,
    };

    void launch_segments(cudaStream_t stream, std::uint32_t spring_count, const float* position_x, const float* position_y, const float* position_z, const std::uint32_t* spring_first, const std::uint32_t* spring_second, const float* rest_lengths, float width, float strain_range, SegmentStyle style, void* output);

    void launch_load_arrow(cudaStream_t stream, float origin_x, float origin_y, float origin_z, float acceleration, float scale, float width, void* output);

} // namespace xayah::cloth::examples::forward::project_cuda

#endif
