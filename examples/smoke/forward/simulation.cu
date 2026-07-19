#include "simulation.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace xayah::smoke::examples::forward::simulation_cuda {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        constexpr float two_pi = 6.28318530717958647692F;

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

        __device__ float gaussian(const float x, const float y, const float z, const Vector center, const float radius) {
            const float dx = x - center.x;
            const float dy = y - center.y;
            const float dz = z - center.z;
            return expf(-0.5F * (dx * dx + dy * dy + dz * dz) / (radius * radius));
        }

        __device__ std::uint64_t index3(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z, const std::uint32_t nx, const std::uint32_t ny) {
            return x + static_cast<std::uint64_t>(nx) * (y + static_cast<std::uint64_t>(ny) * z);
        }

        __global__ void write_control_kernel(const Grid grid, const std::uint64_t step, const float pulse_period, const Vector left_center, const Vector right_center, const float source_radius, const float density_source_rate, const float temperature_source_rate, const Vector left_acceleration, const Vector right_acceleration, float* const density_source, float* const temperature_source, const VectorField external_acceleration) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
            if (cell >= cell_count) return;
            const std::uint32_t x_index = cell % grid.nx;
            const std::uint32_t y_index = cell / grid.nx % grid.ny;
            const std::uint32_t z_index = cell / (grid.nx * grid.ny);
            const float x = (static_cast<float>(x_index) + 0.5F) * grid.cell_size;
            const float y = (static_cast<float>(y_index) + 0.5F) * grid.cell_size;
            const float z = (static_cast<float>(z_index) + 0.5F) * grid.cell_size;
            const float time = static_cast<float>(step) * grid.time_step;
            const float phase = two_pi * time / pulse_period;
            const float left_amplitude = 0.75F + 0.25F * sinf(phase);
            const float right_amplitude = 0.75F + 0.25F * sinf(phase + 0.5F * two_pi);
            const float left_weight = left_amplitude * gaussian(x, y, z, left_center, source_radius);
            const float right_weight = right_amplitude * gaussian(x, y, z, right_center, source_radius);
            const float source_weight = left_weight + right_weight;
            density_source[cell] = density_source_rate * source_weight;
            temperature_source[cell] = temperature_source_rate * source_weight;
            external_acceleration.x[cell] = left_weight * left_acceleration.x + right_weight * right_acceleration.x;
            external_acceleration.y[cell] = left_weight * left_acceleration.y + right_weight * right_acceleration.y;
            external_acceleration.z[cell] = left_weight * left_acceleration.z + right_weight * right_acceleration.z;
        }

        __global__ void reduce_metrics_kernel(const Grid grid, const std::uint32_t* const cell_mask, const float* const density, const float* const temperature, const ConstStaggeredVectorField pre_projection_velocity, const ConstStaggeredVectorField post_projection_velocity, double* const metrics) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
            if (cell >= cell_count || cell_mask[cell] != 0u) return;
            const std::uint32_t x = cell % grid.nx;
            const std::uint32_t y = cell / grid.nx % grid.ny;
            const std::uint32_t z = cell / (grid.nx * grid.ny);
            const std::uint64_t x0 = index3(x, y, z, grid.nx + 1u, grid.ny);
            const std::uint64_t x1 = index3(x + 1u, y, z, grid.nx + 1u, grid.ny);
            const std::uint64_t y0 = index3(x, y, z, grid.nx, grid.ny + 1u);
            const std::uint64_t y1 = index3(x, y + 1u, z, grid.nx, grid.ny + 1u);
            const std::uint64_t z0 = index3(x, y, z, grid.nx, grid.ny);
            const std::uint64_t z1 = index3(x, y, z + 1u, grid.nx, grid.ny);
            const double velocity_x = 0.5 * static_cast<double>(post_projection_velocity.x[x0] + post_projection_velocity.x[x1]);
            const double velocity_y = 0.5 * static_cast<double>(post_projection_velocity.y[y0] + post_projection_velocity.y[y1]);
            const double velocity_z = 0.5 * static_cast<double>(post_projection_velocity.z[z0] + post_projection_velocity.z[z1]);
            const double maximum_velocity = sqrt(velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z);
            const double inverse_spacing = 1.0 / grid.cell_size;
            const double pre_divergence = (pre_projection_velocity.x[x1] - pre_projection_velocity.x[x0] + pre_projection_velocity.y[y1] - pre_projection_velocity.y[y0] + pre_projection_velocity.z[z1] - pre_projection_velocity.z[z0]) * inverse_spacing;
            const double post_divergence = (post_projection_velocity.x[x1] - post_projection_velocity.x[x0] + post_projection_velocity.y[y1] - post_projection_velocity.y[y0] + post_projection_velocity.z[z1] - post_projection_velocity.z[z0]) * inverse_spacing;
            atomicAdd(metrics, static_cast<double>(density[cell]));
            atomic_max(metrics + 1u, density[cell]);
            atomic_max(metrics + 2u, temperature[cell]);
            atomic_max(metrics + 3u, maximum_velocity);
            atomicAdd(metrics + 4u, pre_divergence * pre_divergence);
            atomicAdd(metrics + 5u, post_divergence * post_divergence);
        }

        std::uint32_t blocks(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }

        void check_launch() {
            if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        }

    } // namespace

    void launch_write_control(const cudaStream_t stream, const Grid grid, const std::uint64_t step, const float pulse_period, const Vector left_center, const Vector right_center, const float source_radius, const float density_source_rate, const float temperature_source_rate, const Vector left_acceleration, const Vector right_acceleration, float* const density_source, float* const temperature_source, const VectorField external_acceleration) {
        const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
        write_control_kernel<<<blocks(cell_count), block_size, 0u, stream>>>(grid, step, pulse_period, left_center, right_center, source_radius, density_source_rate, temperature_source_rate, left_acceleration, right_acceleration, density_source, temperature_source, external_acceleration);
        check_launch();
    }

    void launch_reduce_metrics(const cudaStream_t stream, const Grid grid, const std::uint32_t* const cell_mask, const float* const density, const float* const temperature, const ConstStaggeredVectorField pre_projection_velocity, const ConstStaggeredVectorField post_projection_velocity, double* const metrics) {
        const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
        reduce_metrics_kernel<<<blocks(cell_count), block_size, 0u, stream>>>(grid, cell_mask, density, temperature, pre_projection_velocity, post_projection_velocity, metrics);
        check_launch();
    }

} // namespace xayah::smoke::examples::forward::simulation_cuda
