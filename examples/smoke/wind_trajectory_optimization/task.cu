#include "task.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace xayah::smoke::examples::wind_trajectory_optimization::task_cuda {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        constexpr std::uint32_t keyframe_count = 6u;
        constexpr float source_center_x = 0.5F;
        constexpr float source_center_y = 0.078125F;
        constexpr float source_center_z = 0.5F;
        constexpr float source_radius = 0.08F;
        constexpr float wind_center_x = 0.5F;
        constexpr float wind_center_y = 0.75F;
        constexpr float wind_center_z = 0.5F;
        constexpr float wind_radius = 0.35F;

        __device__ void interpolation(const std::uint32_t control_step, const std::uint32_t trajectory_steps, std::uint32_t& first, std::uint32_t& second, float& weight) {
            const float coordinate = static_cast<float>(control_step) * static_cast<float>(keyframe_count - 1u) / static_cast<float>(trajectory_steps - 1u);
            first = min(static_cast<std::uint32_t>(coordinate), keyframe_count - 2u);
            second = first + 1u;
            weight = coordinate - static_cast<float>(first);
        }

        __device__ double block_sum(double value) {
            __shared__ double values[block_size];
            values[threadIdx.x] = value;
            __syncthreads();
            for (std::uint32_t offset = block_size / 2u; offset != 0u; offset /= 2u) {
                if (threadIdx.x < offset) values[threadIdx.x] += values[threadIdx.x + offset];
                __syncthreads();
            }
            return values[0];
        }

        __device__ float gaussian(const float x, const float y, const float z, const float center_x, const float center_y, const float center_z, const float radius) {
            const float dx = x - center_x;
            const float dy = y - center_y;
            const float dz = z - center_z;
            return expf(-0.5F * (dx * dx + dy * dy + dz * dz) / (radius * radius));
        }

        __global__ void write_control_kernel(const Grid grid, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const float* const keyframes, const float density_source_rate, const float temperature_source_rate, float* const density_source, float* const temperature_source, const VectorField external_acceleration) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
            if (cell >= cell_count) return;
            const std::uint32_t x_index = cell % grid.nx;
            const std::uint32_t y_index = cell / grid.nx % grid.ny;
            const std::uint32_t z_index = cell / (grid.nx * grid.ny);
            const float x = (static_cast<float>(x_index) + 0.5F) * grid.cell_size;
            const float y = (static_cast<float>(y_index) + 0.5F) * grid.cell_size;
            const float z = (static_cast<float>(z_index) + 0.5F) * grid.cell_size;
            const float source_weight = gaussian(x, y, z, source_center_x, source_center_y, source_center_z, source_radius);
            const float wind_weight = gaussian(x, y, z, wind_center_x, wind_center_y, wind_center_z, wind_radius);
            std::uint32_t first;
            std::uint32_t second;
            float weight;
            interpolation(control_step, trajectory_steps, first, second, weight);
            density_source[cell] = density_source_rate * source_weight;
            temperature_source[cell] = temperature_source_rate * source_weight;
            external_acceleration.x[cell] = wind_weight * ((1.0F - weight) * keyframes[2u * first] + weight * keyframes[2u * second]);
            external_acceleration.y[cell] = 0.0F;
            external_acceleration.z[cell] = wind_weight * ((1.0F - weight) * keyframes[2u * first + 1u] + weight * keyframes[2u * second + 1u]);
        }

        __global__ void density_energy_kernel(const std::uint32_t cell_count, const float* const density, double* const energy) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            const double value = cell < cell_count ? static_cast<double>(density[cell]) * density[cell] : 0.0;
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(energy, sum);
        }

        __global__ void density_loss_kernel(const std::uint32_t cell_count, const double inverse_target_frame_energy, const float* const density, const float* const target_density, double* const loss) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            double value = 0.0;
            if (cell < cell_count) {
                const double difference = static_cast<double>(density[cell] - target_density[cell]);
                value = 0.5 * inverse_target_frame_energy * difference * difference;
            }
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(loss, sum);
        }

        __global__ void density_loss_seed_kernel(const std::uint32_t cell_count, const double inverse_target_frame_energy, const float* const density, const float* const target_density, double* const density_seed, double* const loss) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            double value = 0.0;
            if (cell < cell_count) {
                const float difference = density[cell] - target_density[cell];
                density_seed[cell] = inverse_target_frame_energy * difference;
                value = 0.5 * inverse_target_frame_energy * static_cast<double>(difference) * difference;
            }
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(loss, sum);
        }

        __global__ void density_tangent_inner_product_kernel(const std::uint32_t cell_count, const double* const density_seed, const float* const density_tangent, double* const result) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            const double value = cell < cell_count ? static_cast<double>(density_seed[cell]) * density_tangent[cell] : 0.0;
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(result, sum);
        }

        __global__ void accumulate_keyframe_gradient_kernel(const Grid grid, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const ConstAdjointVectorField acceleration_adjoint, double* const keyframe_gradient) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
            if (cell >= cell_count) return;
            const std::uint32_t x_index = cell % grid.nx;
            const std::uint32_t y_index = cell / grid.nx % grid.ny;
            const std::uint32_t z_index = cell / (grid.nx * grid.ny);
            const float x = (static_cast<float>(x_index) + 0.5F) * grid.cell_size;
            const float y = (static_cast<float>(y_index) + 0.5F) * grid.cell_size;
            const float z = (static_cast<float>(z_index) + 0.5F) * grid.cell_size;
            const float spatial_weight = gaussian(x, y, z, wind_center_x, wind_center_y, wind_center_z, wind_radius);
            std::uint32_t first;
            std::uint32_t second;
            float weight;
            interpolation(control_step, trajectory_steps, first, second, weight);
            atomicAdd(keyframe_gradient + 2u * first, static_cast<double>(spatial_weight * (1.0F - weight)) * acceleration_adjoint.x[cell]);
            atomicAdd(keyframe_gradient + 2u * first + 1u, static_cast<double>(spatial_weight * (1.0F - weight)) * acceleration_adjoint.z[cell]);
            atomicAdd(keyframe_gradient + 2u * second, static_cast<double>(spatial_weight * weight) * acceleration_adjoint.x[cell]);
            atomicAdd(keyframe_gradient + 2u * second + 1u, static_cast<double>(spatial_weight * weight) * acceleration_adjoint.z[cell]);
        }

        std::uint32_t grid_size(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }

        void check_launch() {
            if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        }

    } // namespace

    void launch_write_control(const cudaStream_t stream, const Grid grid, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const float* const keyframes, const float density_source_rate, const float temperature_source_rate, float* const density_source, float* const temperature_source, const VectorField external_acceleration) {
        const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
        write_control_kernel<<<grid_size(cell_count), block_size, 0u, stream>>>(grid, control_step, trajectory_steps, keyframes, density_source_rate, temperature_source_rate, density_source, temperature_source, external_acceleration);
        check_launch();
    }

    void launch_write_control_tangent(const cudaStream_t stream, const Grid grid, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const float* const keyframes, float* const density_source, float* const temperature_source, const VectorField external_acceleration) {
        launch_write_control(stream, grid, control_step, trajectory_steps, keyframes, 0.0F, 0.0F, density_source, temperature_source, external_acceleration);
    }

    void launch_density_energy(const cudaStream_t stream, const std::uint32_t cell_count, const float* const density, double* const energy) {
        density_energy_kernel<<<grid_size(cell_count), block_size, 0u, stream>>>(cell_count, density, energy);
        check_launch();
    }

    void launch_density_loss(const cudaStream_t stream, const std::uint32_t cell_count, const double inverse_target_frame_energy, const float* const density, const float* const target_density, double* const loss) {
        density_loss_kernel<<<grid_size(cell_count), block_size, 0u, stream>>>(cell_count, inverse_target_frame_energy, density, target_density, loss);
        check_launch();
    }

    void launch_density_loss_seed(const cudaStream_t stream, const std::uint32_t cell_count, const double inverse_target_frame_energy, const float* const density, const float* const target_density, double* const density_seed, double* const loss) {
        density_loss_seed_kernel<<<grid_size(cell_count), block_size, 0u, stream>>>(cell_count, inverse_target_frame_energy, density, target_density, density_seed, loss);
        check_launch();
    }

    void launch_density_tangent_inner_product(const cudaStream_t stream, const std::uint32_t cell_count, const double* const density_seed, const float* const density_tangent, double* const result) {
        density_tangent_inner_product_kernel<<<grid_size(cell_count), block_size, 0u, stream>>>(cell_count, density_seed, density_tangent, result);
        check_launch();
    }

    void launch_accumulate_keyframe_gradient(const cudaStream_t stream, const Grid grid, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const ConstAdjointVectorField acceleration_adjoint, double* const keyframe_gradient) {
        const std::uint32_t cell_count = grid.nx * grid.ny * grid.nz;
        accumulate_keyframe_gradient_kernel<<<grid_size(cell_count), block_size, 0u, stream>>>(grid, control_step, trajectory_steps, acceleration_adjoint, keyframe_gradient);
        check_launch();
    }

} // namespace xayah::smoke::examples::wind_trajectory_optimization::task_cuda
