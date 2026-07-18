#include "task.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace xayah::cloth::examples::stretch_stiffness_inverse::inverse_cuda {

    namespace {

        constexpr std::uint32_t block_size = 256u;

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

        __global__ void fill_kernel(const std::uint32_t count, const float value, float* const output) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < count) output[index] = value;
        }

        __global__ void position_loss_kernel(const std::uint32_t particle_count, const double normalization, const ConstField positions, const ConstField target_positions, double* const loss) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            double value                 = 0.0;
            if (particle < particle_count) {
                const double x = static_cast<double>(positions.x[particle] - target_positions.x[particle]);
                const double y = static_cast<double>(positions.y[particle] - target_positions.y[particle]);
                const double z = static_cast<double>(positions.z[particle] - target_positions.z[particle]);
                value          = 0.5 * normalization * (x * x + y * y + z * z);
            }
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(loss, sum);
        }

        __global__ void position_loss_seed_kernel(const std::uint32_t particle_count, const float normalization, const ConstField positions, const ConstField target_positions, const Field position_seed, double* const loss) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            double value                 = 0.0;
            if (particle < particle_count) {
                const float x             = positions.x[particle] - target_positions.x[particle];
                const float y             = positions.y[particle] - target_positions.y[particle];
                const float z             = positions.z[particle] - target_positions.z[particle];
                position_seed.x[particle] = normalization * x;
                position_seed.y[particle] = normalization * y;
                position_seed.z[particle] = normalization * z;
                value                     = 0.5 * static_cast<double>(normalization) * (static_cast<double>(x) * x + static_cast<double>(y) * y + static_cast<double>(z) * z);
            }
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(loss, sum);
        }

        __global__ void position_tangent_inner_product_kernel(const std::uint32_t particle_count, const ConstField position_seed, const ConstField position_tangent, double* const result) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            double value                 = 0.0;
            if (particle < particle_count)
                value = static_cast<double>(position_seed.x[particle]) * position_tangent.x[particle] + static_cast<double>(position_seed.y[particle]) * position_tangent.y[particle] + static_cast<double>(position_seed.z[particle]) * position_tangent.z[particle];
            const double sum = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(result, sum);
        }

        __global__ void sum_kernel(const std::uint32_t count, const float* const values, double* const result) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const double value        = index < count ? static_cast<double>(values[index]) : 0.0;
            const double sum          = block_sum(value);
            if (threadIdx.x == 0u) atomicAdd(result, sum);
        }

        std::uint32_t grid_size(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }

    } // namespace

    void launch_fill(const cudaStream_t stream, const std::uint32_t count, const float value, float* const output) {
        if (count == 0u) return;
        fill_kernel<<<grid_size(count), block_size, 0u, stream>>>(count, value, output);
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

    void launch_sum(const cudaStream_t stream, const std::uint32_t count, const float* const values, double* const result) {
        if (count == 0u) return;
        sum_kernel<<<grid_size(count), block_size, 0u, stream>>>(count, values, result);
        if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

} // namespace xayah::cloth::examples::stretch_stiffness_inverse::inverse_cuda
