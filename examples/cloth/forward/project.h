#ifndef XAYAH_EXAMPLES_CLOTH_FORWARD_PROJECT_H
#define XAYAH_EXAMPLES_CLOTH_FORWARD_PROJECT_H

#include <cstdint>
#include <cuda_runtime_api.h>
#include <spectra/sdk/cuda_types.h>

namespace xayah::cloth::examples::forward::project_cuda {
    enum class SegmentStyle : std::uint32_t {
        Stretch,
        Bending,
    };

    void launch_positions(cudaStream_t stream, std::uint32_t vertex_count, const float* position_x, const float* position_y, const float* position_z, spectra::sdk::Float3* output);

    void launch_segments(cudaStream_t stream, std::uint32_t spring_count, const float* position_x, const float* position_y, const float* position_z, const std::uint32_t* spring_first, const std::uint32_t* spring_second, const float* rest_lengths, float width, float strain_range, SegmentStyle style, spectra::sdk::Line* output);

    void launch_wind_vector(cudaStream_t stream, float origin_x, float origin_y, float origin_z, float wind_x, float wind_z, float scale, float width, spectra::sdk::Vector* output);

} // namespace xayah::cloth::examples::forward::project_cuda

#endif
