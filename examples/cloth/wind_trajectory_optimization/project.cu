#include "project.h"

#include <cuda_runtime.h>

namespace xayah::cloth::examples::wind_trajectory_optimization::visualization_cuda {

    namespace {

        struct SegmentInstance {
            float sx;
            float sy;
            float sz;
            float width;
            float ex;
            float ey;
            float ez;
            std::uint32_t flags;
            float r;
            float g;
            float b;
            float a;
        };

        static_assert(sizeof(SegmentInstance) == 48u);

        __global__ void write_segments(
            const std::uint32_t spring_count,
            const float* const position_x,
            const float* const position_y,
            const float* const position_z,
            const std::uint32_t* const spring_first,
            const std::uint32_t* const spring_second,
            const float* const rest_lengths,
            const float width,
            const float strain_range,
            const SegmentStyle style,
            SegmentInstance* const output) {
            const std::uint32_t spring_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (spring_index >= spring_count) return;

            const std::uint32_t first = spring_first[spring_index];
            const std::uint32_t second = spring_second[spring_index];
            const float sx = position_x[first];
            const float sy = position_y[first];
            const float sz = position_z[first];
            const float ex = position_x[second];
            const float ey = position_y[second];
            const float ez = position_z[second];
            const float dx = ex - sx;
            const float dy = ey - sy;
            const float dz = ez - sz;
            const float strain = (sqrtf(dx * dx + dy * dy + dz * dz) - rest_lengths[spring_index]) / rest_lengths[spring_index];
            const float normalized = fminf(fmaxf(strain / strain_range, -1.0F), 1.0F);

            SegmentInstance instance{
                .sx = sx,
                .sy = sy,
                .sz = sz,
                .width = width,
                .ex = ex,
                .ey = ey,
                .ez = ez,
                .flags = 0u,
            };
            if (style == SegmentStyle::target) {
                instance.r = 0.10F;
                instance.g = 0.84F;
                instance.b = 0.96F;
                instance.a = 0.35F;
            } else if (style == SegmentStyle::bending) {
                instance.r = 0.64F;
                instance.g = 0.32F;
                instance.b = 0.92F;
                instance.a = 0.46F;
            } else if (normalized < 0.0F) {
                const float amount = -normalized;
                instance.r = 0.78F - 0.62F * amount;
                instance.g = 0.80F - 0.36F * amount;
                instance.b = 0.82F + 0.16F * amount;
                instance.a = 1.0F;
            } else {
                instance.r = 0.78F + 0.20F * normalized;
                instance.g = 0.80F - 0.64F * normalized;
                instance.b = 0.82F - 0.66F * normalized;
                instance.a = 1.0F;
            }
            output[spring_index] = instance;
        }

        __global__ void write_wind_arrow(const float origin_x, const float origin_y, const float origin_z, const float wind_x, const float wind_z, const float scale, const float width, const SegmentStyle style, SegmentInstance* const output) {
            const float vector_x = scale * wind_x;
            const float vector_z = scale * wind_z;
            const float end_x = origin_x + vector_x;
            const float end_z = origin_z + vector_z;
            const float length = sqrtf(vector_x * vector_x + vector_z * vector_z);
            float direction_x = 0.0F;
            float direction_z = 0.0F;
            if (length > 1.0e-6F) {
                direction_x = vector_x / length;
                direction_z = vector_z / length;
            }
            const float head_length = 0.28F * length;
            const float head_width = 0.16F * length;
            const float base_x = end_x - head_length * direction_x;
            const float base_z = end_z - head_length * direction_z;
            const float perpendicular_x = -direction_z;
            const float perpendicular_z = direction_x;
            const float red = style == SegmentStyle::target_wind ? 0.10F : 1.00F;
            const float green = style == SegmentStyle::target_wind ? 0.84F : 0.48F;
            const float blue = style == SegmentStyle::target_wind ? 0.96F : 0.08F;
            const float alpha = style == SegmentStyle::target_wind ? 0.72F : 1.00F;
            output[0] = {.sx = origin_x, .sy = origin_y, .sz = origin_z, .width = width, .ex = end_x, .ey = origin_y, .ez = end_z, .flags = 0u, .r = red, .g = green, .b = blue, .a = alpha};
            output[1] = {.sx = end_x, .sy = origin_y, .sz = end_z, .width = width, .ex = base_x + head_width * perpendicular_x, .ey = origin_y, .ez = base_z + head_width * perpendicular_z, .flags = 0u, .r = red, .g = green, .b = blue, .a = alpha};
            output[2] = {.sx = end_x, .sy = origin_y, .sz = end_z, .width = width, .ex = base_x - head_width * perpendicular_x, .ey = origin_y, .ez = base_z - head_width * perpendicular_z, .flags = 0u, .r = red, .g = green, .b = blue, .a = alpha};
        }

    } // namespace

    void launch_segments(
        const cudaStream_t stream,
        const std::uint32_t spring_count,
        const float* const position_x,
        const float* const position_y,
        const float* const position_z,
        const std::uint32_t* const spring_first,
        const std::uint32_t* const spring_second,
        const float* const rest_lengths,
        const float width,
        const float strain_range,
        const SegmentStyle style,
        void* const output) {
        if (spring_count == 0u) return;
        write_segments<<<(spring_count + 255u) / 256u, 256u, 0u, stream>>>(spring_count, position_x, position_y, position_z, spring_first, spring_second, rest_lengths, width, strain_range, style, static_cast<SegmentInstance*>(output));
    }

    void launch_wind_arrow(const cudaStream_t stream, const float origin_x, const float origin_y, const float origin_z, const float wind_x, const float wind_z, const float scale, const float width, const SegmentStyle style, void* const output) {
        write_wind_arrow<<<1u, 1u, 0u, stream>>>(origin_x, origin_y, origin_z, wind_x, wind_z, scale, width, style, static_cast<SegmentInstance*>(output));
    }

} // namespace xayah::cloth::examples::wind_trajectory_optimization::visualization_cuda
