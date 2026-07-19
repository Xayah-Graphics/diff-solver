#ifndef XAYAH_EXAMPLES_CLOTH_WIND_TRAJECTORY_OPTIMIZATION_PROJECT_H
#define XAYAH_EXAMPLES_CLOTH_WIND_TRAJECTORY_OPTIMIZATION_PROJECT_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::cloth::examples::wind_trajectory_optimization::project_cuda {

    enum class SegmentStyle : std::uint32_t {
        estimate,
        target,
        bending,
        target_wind,
        estimated_wind,
    };

    void launch_segments(cudaStream_t stream, std::uint32_t spring_count, const float* position_x, const float* position_y, const float* position_z, const std::uint32_t* spring_first, const std::uint32_t* spring_second, const float* rest_lengths, float width, float strain_range, SegmentStyle style, void* output);

    void launch_wind_arrow(cudaStream_t stream, float origin_x, float origin_y, float origin_z, float wind_x, float wind_z, float scale, float width, SegmentStyle style, void* output);

} // namespace xayah::cloth::examples::wind_trajectory_optimization::project_cuda

#endif
