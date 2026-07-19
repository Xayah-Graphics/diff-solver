#include "project.h"

#include <cuda_runtime.h>

namespace xayah::smoke::examples::wind_trajectory_optimization::visualization_cuda {

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

        __device__ double atomic_max(double* const address, const double value) {
            unsigned long long* const integer_address = reinterpret_cast<unsigned long long*>(address);
            unsigned long long previous = *integer_address;
            while (__longlong_as_double(previous) < value) {
                const unsigned long long assumed = previous;
                previous = atomicCAS(integer_address, assumed, __double_as_longlong(value));
                if (previous == assumed) break;
            }
            return __longlong_as_double(previous);
        }

        __global__ void density_statistics(const std::uint32_t cell_count, const float* const target_density, const float* const estimated_density, double* const statistics) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            if (cell >= cell_count) return;
            const double target = target_density[cell];
            const double estimated = estimated_density[cell];
            atomicAdd(statistics, target);
            atomic_max(statistics + 1u, target);
            atomicAdd(statistics + 2u, estimated);
            atomic_max(statistics + 3u, estimated);
        }

        __global__ void write_volume(const std::uint32_t nx, const std::uint32_t ny, const std::uint32_t nz, const float* const target_density, const float* const estimated_density, const bool show_target, const bool show_estimate, const bool show_difference, const float smoke_opacity_scale, const float difference_opacity_scale, float* const density, float* const color) {
            const std::uint32_t output_nx = 3u * nx;
            const std::uint32_t output_cell = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t output_cell_count = output_nx * ny * nz;
            if (output_cell >= output_cell_count) return;
            const std::uint32_t x = output_cell % output_nx;
            const std::uint32_t y = output_cell / output_nx % ny;
            const std::uint32_t z = output_cell / (output_nx * ny);
            const std::uint32_t column = x / nx;
            const std::uint32_t source_x = x % nx;
            const std::uint32_t source_cell = source_x + nx * (y + ny * z);
            const bool visible = column == 0u ? show_target : column == 1u ? show_estimate : show_difference;
            const float value = column == 0u ? target_density[source_cell] : column == 1u ? estimated_density[source_cell] : fabsf(target_density[source_cell] - estimated_density[source_cell]);
            density[output_cell] = visible ? (column == 2u ? difference_opacity_scale : smoke_opacity_scale) * value : 0.0F;
            color[3u * output_cell] = column == 0u ? 0.08F : 1.00F;
            color[3u * output_cell + 1u] = column == 0u ? 0.95F : column == 1u ? 0.48F : 0.06F;
            color[3u * output_cell + 2u] = column == 0u ? 1.00F : column == 1u ? 0.04F : 0.72F;
        }

        __global__ void write_wind_arrow(const float origin_x, const float origin_y, const float origin_z, const float wind_x, const float wind_z, const float scale, const float width, const WindStyle style, SegmentInstance* const output) {
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
            const float red = style == WindStyle::target ? 0.10F : 1.00F;
            const float green = style == WindStyle::target ? 0.84F : 0.48F;
            const float blue = style == WindStyle::target ? 0.96F : 0.08F;
            const float alpha = style == WindStyle::target ? 0.72F : 1.00F;
            output[0] = {.sx = origin_x, .sy = origin_y, .sz = origin_z, .width = width, .ex = end_x, .ey = origin_y, .ez = end_z, .flags = 0u, .r = red, .g = green, .b = blue, .a = alpha};
            output[1] = {.sx = end_x, .sy = origin_y, .sz = end_z, .width = width, .ex = base_x + head_width * perpendicular_x, .ey = origin_y, .ez = base_z + head_width * perpendicular_z, .flags = 0u, .r = red, .g = green, .b = blue, .a = alpha};
            output[2] = {.sx = end_x, .sy = origin_y, .sz = end_z, .width = width, .ex = base_x - head_width * perpendicular_x, .ey = origin_y, .ez = base_z - head_width * perpendicular_z, .flags = 0u, .r = red, .g = green, .b = blue, .a = alpha};
        }

    } // namespace

    void launch_volume(const cudaStream_t stream, const std::uint32_t nx, const std::uint32_t ny, const std::uint32_t nz, const float* const target_density, const float* const estimated_density, const bool show_target, const bool show_estimate, const bool show_difference, const float smoke_opacity_scale, const float difference_opacity_scale, float* const density, float* const color, double* const statistics) {
        const std::uint32_t source_cell_count = nx * ny * nz;
        const std::uint32_t output_cell_count = 3u * source_cell_count;
        density_statistics<<<(source_cell_count + 255u) / 256u, 256u, 0u, stream>>>(source_cell_count, target_density, estimated_density, statistics);
        write_volume<<<(output_cell_count + 255u) / 256u, 256u, 0u, stream>>>(nx, ny, nz, target_density, estimated_density, show_target, show_estimate, show_difference, smoke_opacity_scale, difference_opacity_scale, density, color);
    }

    void launch_wind_arrow(const cudaStream_t stream, const float origin_x, const float origin_y, const float origin_z, const float wind_x, const float wind_z, const float scale, const float width, const WindStyle style, void* const output) {
        write_wind_arrow<<<1u, 1u, 0u, stream>>>(origin_x, origin_y, origin_z, wind_x, wind_z, scale, width, style, static_cast<SegmentInstance*>(output));
    }

} // namespace xayah::smoke::examples::wind_trajectory_optimization::visualization_cuda
