#ifndef XAYAH_EXAMPLES_SMOKE_FORWARD_PROJECT_H
#define XAYAH_EXAMPLES_SMOKE_FORWARD_PROJECT_H

#include <cstdint>
#include <cuda_runtime_api.h>
#include <spectra/sdk/cuda_types.h>

namespace xayah::smoke::examples::forward::project_cuda {
    void launch_volume(cudaStream_t stream, std::uint64_t cell_count, const float* density_source, const float* temperature_source, float density_scale, float* density_output, float* temperature_output);
    void launch_emitter_vector(cudaStream_t stream, float origin_x, float origin_y, float origin_z, float acceleration_x, float acceleration_y, float acceleration_z, float scale, float width, float red, float green, float blue, spectra::sdk::Vector* output);
}

#endif
