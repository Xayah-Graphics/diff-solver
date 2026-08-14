#include "project.h"

#include <cuda_runtime.h>

namespace xayah::smoke::examples::forward::project_cuda {
    namespace {
        __global__ void write_volume(const std::uint64_t cell_count, const float* const density_source, const float* const temperature_source, const float density_scale, float* const density_output, float* const temperature_output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < cell_count) {
                density_output[index] = density_scale * density_source[index];
                temperature_output[index] = temperature_source[index];
            }
        }

        __global__ void write_emitter_vector(const float origin_x, const float origin_y, const float origin_z, const float acceleration_x, const float acceleration_y, const float acceleration_z, const float scale, const float width, const float red, const float green, const float blue, spectra::sdk::Vector* const output) {
            output[0] = {
                .origin = {origin_x, origin_y, origin_z},
                .width  = width,
                .vector = {scale * acceleration_x, scale * acceleration_y, scale * acceleration_z},
                .color  = {red, green, blue, 1.0F},
            };
        }
    }

    void launch_volume(const cudaStream_t stream, const std::uint64_t cell_count, const float* const density_source, const float* const temperature_source, const float density_scale, float* const density_output, float* const temperature_output) {
        constexpr std::uint32_t block_size = 256u;
        write_volume<<<static_cast<std::uint32_t>((cell_count + block_size - 1u) / block_size), block_size, 0u, stream>>>(cell_count, density_source, temperature_source, density_scale, density_output, temperature_output);
    }

    void launch_emitter_vector(const cudaStream_t stream, const float origin_x, const float origin_y, const float origin_z, const float acceleration_x, const float acceleration_y, const float acceleration_z, const float scale, const float width, const float red, const float green, const float blue, spectra::sdk::Vector* const output) {
        write_emitter_vector<<<1u, 1u, 0u, stream>>>(origin_x, origin_y, origin_z, acceleration_x, acceleration_y, acceleration_z, scale, width, red, green, blue, output);
    }
}
