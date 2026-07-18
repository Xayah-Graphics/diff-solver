#include "task.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace xayah::cloth::examples::wind_trajectory_optimization::task_cuda {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        constexpr std::uint32_t keyframe_count = 6u;

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

        __global__ void write_control_kernel(const std::uint32_t particle_count, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const std::uint32_t* const anchor_mask, const float* const keyframes, const Field external_forces) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            if (anchor_mask[particle] != 0u) {
                external_forces.x[particle] = 0.0F;
                external_forces.y[particle] = 0.0F;
                external_forces.z[particle] = 0.0F;
                return;
            }
            std::uint32_t first;
            std::uint32_t second;
            float weight;
            interpolation(control_step, trajectory_steps, first, second, weight);
            external_forces.x[particle] = (1.0F - weight) * keyframes[2u * first] + weight * keyframes[2u * second];
            external_forces.y[particle] = 0.0F;
            external_forces.z[particle] = (1.0F - weight) * keyframes[2u * first + 1u] + weight * keyframes[2u * second + 1u];
        }

        __global__ void position_loss_kernel(const std::uint32_t particle_count, const double normalization, const ConstField positions, const ConstField target_positions, double* const loss) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            double value = 0.0;
            if (particle < particle_count) {
                const double x = static_cast<double>(positions.x[particle] - target_positions.x[particle]);
                const double y = static_cast<double>(positions.y[particle] - target_positions.y[particle]);
                const double z = static_cast<double>(positions.z[particle] - target_positions.z[particle]);
                value = 0.5 * normalization * (x * x + y * y + z * z);
            }
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(loss, sum);
        }

        __global__ void position_loss_seed_kernel(const std::uint32_t particle_count, const float normalization, const ConstField positions, const ConstField target_positions, const Field position_seed, double* const loss) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            double value = 0.0;
            if (particle < particle_count) {
                const float x = positions.x[particle] - target_positions.x[particle];
                const float y = positions.y[particle] - target_positions.y[particle];
                const float z = positions.z[particle] - target_positions.z[particle];
                position_seed.x[particle] = normalization * x;
                position_seed.y[particle] = normalization * y;
                position_seed.z[particle] = normalization * z;
                value = 0.5 * static_cast<double>(normalization) * (static_cast<double>(x) * x + static_cast<double>(y) * y + static_cast<double>(z) * z);
            }
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(loss, sum);
        }

        __global__ void position_tangent_inner_product_kernel(const std::uint32_t particle_count, const ConstField position_seed, const ConstField position_tangent, double* const result) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            double value = 0.0;
            if (particle < particle_count)
                value = static_cast<double>(position_seed.x[particle]) * position_tangent.x[particle] + static_cast<double>(position_seed.y[particle]) * position_tangent.y[particle] + static_cast<double>(position_seed.z[particle]) * position_tangent.z[particle];
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(result, sum);
        }

        __global__ void accumulate_keyframe_gradient_kernel(const std::uint32_t particle_count, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const std::uint32_t* const anchor_mask, const ConstField control_adjoint, double* const keyframe_gradient) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count || anchor_mask[particle] != 0u) return;
            std::uint32_t first;
            std::uint32_t second;
            float weight;
            interpolation(control_step, trajectory_steps, first, second, weight);
            atomicAdd(keyframe_gradient + 2u * first, static_cast<double>(1.0F - weight) * control_adjoint.x[particle]);
            atomicAdd(keyframe_gradient + 2u * first + 1u, static_cast<double>(1.0F - weight) * control_adjoint.z[particle]);
            atomicAdd(keyframe_gradient + 2u * second, static_cast<double>(weight) * control_adjoint.x[particle]);
            atomicAdd(keyframe_gradient + 2u * second + 1u, static_cast<double>(weight) * control_adjoint.z[particle]);
        }

        std::uint32_t grid_size(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }

    } // namespace

    void launch_write_control(const cudaStream_t stream, const std::uint32_t particle_count, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const std::uint32_t* const anchor_mask, const float* const keyframes, const Field external_forces) {
        write_control_kernel<<<grid_size(particle_count), block_size, 0u, stream>>>(particle_count, control_step, trajectory_steps, anchor_mask, keyframes, external_forces);
        if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void launch_position_loss(const cudaStream_t stream, const std::uint32_t particle_count, const double normalization, const ConstField positions, const ConstField target_positions, double* const loss) {
        position_loss_kernel<<<grid_size(particle_count), block_size, 0u, stream>>>(particle_count, normalization, positions, target_positions, loss);
        if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void launch_position_loss_seed(const cudaStream_t stream, const std::uint32_t particle_count, const float normalization, const ConstField positions, const ConstField target_positions, const Field position_seed, double* const loss) {
        position_loss_seed_kernel<<<grid_size(particle_count), block_size, 0u, stream>>>(particle_count, normalization, positions, target_positions, position_seed, loss);
        if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void launch_position_tangent_inner_product(const cudaStream_t stream, const std::uint32_t particle_count, const ConstField position_seed, const ConstField position_tangent, double* const result) {
        position_tangent_inner_product_kernel<<<grid_size(particle_count), block_size, 0u, stream>>>(particle_count, position_seed, position_tangent, result);
        if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void launch_accumulate_keyframe_gradient(const cudaStream_t stream, const std::uint32_t particle_count, const std::uint32_t control_step, const std::uint32_t trajectory_steps, const std::uint32_t* const anchor_mask, const ConstField control_adjoint, double* const keyframe_gradient) {
        accumulate_keyframe_gradient_kernel<<<grid_size(particle_count), block_size, 0u, stream>>>(particle_count, control_step, trajectory_steps, anchor_mask, control_adjoint, keyframe_gradient);
        if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

} // namespace xayah::cloth::examples::wind_trajectory_optimization::task_cuda
