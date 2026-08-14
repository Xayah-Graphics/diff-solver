#include "project.h"
#include <cuda_runtime.h>

namespace xayah::cloth::examples::forward::project_cuda {

    namespace {

        __global__ void write_positions(const std::uint32_t vertex_count, const float* const position_x, const float* const position_y, const float* const position_z, spectra::sdk::Float3* const output) {
            const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
            if (vertex >= vertex_count) return;
            output[vertex] = {position_x[vertex], position_y[vertex], position_z[vertex]};
        }

        __global__ void write_segments(const std::uint32_t spring_count, const float* const position_x, const float* const position_y, const float* const position_z, const std::uint32_t* const spring_first, const std::uint32_t* const spring_second, const float* const rest_lengths, const float width, const float strain_range, const SegmentStyle style, spectra::sdk::Line* const output) {
            const std::uint32_t spring_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (spring_index >= spring_count) return;

            const std::uint32_t first  = spring_first[spring_index];
            const std::uint32_t second = spring_second[spring_index];
            const float sx             = position_x[first];
            const float sy             = position_y[first];
            const float sz             = position_z[first];
            const float ex             = position_x[second];
            const float ey             = position_y[second];
            const float ez             = position_z[second];
            const float dx             = ex - sx;
            const float dy             = ey - sy;
            const float dz             = ez - sz;
            const float strain         = (sqrtf(dx * dx + dy * dy + dz * dz) - rest_lengths[spring_index]) / rest_lengths[spring_index];
            const float normalized     = fminf(fmaxf(strain / strain_range, -1.0F), 1.0F);

            spectra::sdk::Line instance{
                .first_position  = {sx, sy, sz},
                .width           = width,
                .second_position = {ex, ey, ez},
            };
            if (style == SegmentStyle::Bending) {
                instance.color = {0.64F, 0.32F, 0.92F, 0.46F};
            } else if (normalized < 0.0F) {
                const float amount = -normalized;
                instance.color     = {0.78F - 0.62F * amount, 0.80F - 0.36F * amount, 0.82F + 0.16F * amount, 1.0F};
            } else {
                instance.color = {0.78F + 0.20F * normalized, 0.80F - 0.64F * normalized, 0.82F - 0.66F * normalized, 1.0F};
            }
            output[spring_index] = instance;
        }

        __global__ void write_wind_vector(const float origin_x, const float origin_y, const float origin_z, const float wind_x, const float wind_z, const float scale, const float width, spectra::sdk::Vector* const output) {
            output[0] = {
                .origin = {origin_x, origin_y, origin_z},
                .width  = width,
                .vector = {scale * wind_x, 0.0F, scale * wind_z},
                .color  = {1.00F, 0.48F, 0.08F, 1.00F},
            };
        }

    } // namespace

    void launch_positions(const cudaStream_t stream, const std::uint32_t vertex_count, const float* const position_x, const float* const position_y, const float* const position_z, spectra::sdk::Float3* const output) {
        write_positions<<<(vertex_count + 255u) / 256u, 256u, 0u, stream>>>(vertex_count, position_x, position_y, position_z, output);
    }

    void launch_segments(const cudaStream_t stream, const std::uint32_t spring_count, const float* const position_x, const float* const position_y, const float* const position_z, const std::uint32_t* const spring_first, const std::uint32_t* const spring_second, const float* const rest_lengths, const float width, const float strain_range, const SegmentStyle style, spectra::sdk::Line* const output) {
        if (spring_count == 0u) return;
        write_segments<<<(spring_count + 255u) / 256u, 256u, 0u, stream>>>(spring_count, position_x, position_y, position_z, spring_first, spring_second, rest_lengths, width, strain_range, style, output);
    }

    void launch_wind_vector(const cudaStream_t stream, const float origin_x, const float origin_y, const float origin_z, const float wind_x, const float wind_z, const float scale, const float width, spectra::sdk::Vector* const output) {
        write_wind_vector<<<1u, 1u, 0u, stream>>>(origin_x, origin_y, origin_z, wind_x, wind_z, scale, width, output);
    }

} // namespace xayah::cloth::examples::forward::project_cuda
