#ifndef XAYAH_EXAMPLES_SMOKE_WIND_TRAJECTORY_OPTIMIZATION_PROJECT_H
#define XAYAH_EXAMPLES_SMOKE_WIND_TRAJECTORY_OPTIMIZATION_PROJECT_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::smoke::examples::wind_trajectory_optimization::visualization_cuda {

    enum class WindStyle : std::uint32_t {
        target,
        estimated,
    };

    void launch_volume(cudaStream_t stream, std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, const float* target_density, const float* estimated_density, bool show_target, bool show_estimate, bool show_difference, float smoke_opacity_scale, float difference_opacity_scale, float* density, float* color, double* statistics);
    void launch_wind_arrow(cudaStream_t stream, float origin_x, float origin_y, float origin_z, float wind_x, float wind_z, float scale, float width, WindStyle style, void* output);

} // namespace xayah::smoke::examples::wind_trajectory_optimization::visualization_cuda

#endif
