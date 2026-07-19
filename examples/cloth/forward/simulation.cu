#include "simulation.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace xayah::cloth::examples::forward::simulation_cuda {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        constexpr float two_pi = 6.28318530717958647692F;
        constexpr float one_third_pi = 1.04719755119659774615F;

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

        __device__ float load_acceleration(const float time, const float normalized_x, const float ramp_duration, const float period, const float base_acceleration, const float primary_acceleration, const float secondary_acceleration) {
            const float ramp_coordinate = fminf(time / ramp_duration, 1.0F);
            const float ramp = ramp_coordinate * ramp_coordinate * (3.0F - 2.0F * ramp_coordinate);
            const float spatial = normalized_x * normalized_x * (3.0F - 2.0F * normalized_x);
            const float phase = two_pi * (time / period - normalized_x);
            return ramp * spatial * (base_acceleration + primary_acceleration * sinf(phase) + secondary_acceleration * sinf(2.0F * phase + one_third_pi));
        }

        __global__ void write_control(const std::uint32_t particle_count, const std::uint32_t columns, const std::uint64_t step, const float time_step, const float mass, const float ramp_duration, const float period, const float base_acceleration, const float primary_acceleration, const float secondary_acceleration, const Field external_forces, double* const metrics) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const std::uint32_t column = particle % columns;
            const float normalized_x = static_cast<float>(column) / static_cast<float>(columns - 1u);
            const float acceleration = column == 0u ? 0.0F : load_acceleration(static_cast<float>(step) * time_step, normalized_x, ramp_duration, period, base_acceleration, primary_acceleration, secondary_acceleration);
            external_forces.x[particle] = 0.0F;
            external_forces.y[particle] = 0.0F;
            external_forces.z[particle] = mass * acceleration;
            if (particle != 0u) return;
            metrics[7] = load_acceleration(static_cast<float>(step) * time_step, 0.25F, ramp_duration, period, base_acceleration, primary_acceleration, secondary_acceleration);
            metrics[8] = load_acceleration(static_cast<float>(step) * time_step, 0.50F, ramp_duration, period, base_acceleration, primary_acceleration, secondary_acceleration);
            metrics[9] = load_acceleration(static_cast<float>(step) * time_step, 0.75F, ramp_duration, period, base_acceleration, primary_acceleration, secondary_acceleration);
        }

        __global__ void reduce_particles(const std::uint32_t particle_count, const std::uint32_t columns, const float* const masses, const ConstField positions, const ConstField velocities, double* const metrics) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const double velocity_squared = static_cast<double>(velocities.x[particle]) * velocities.x[particle] + static_cast<double>(velocities.y[particle]) * velocities.y[particle] + static_cast<double>(velocities.z[particle]) * velocities.z[particle];
            atomicAdd(metrics, 0.5 * masses[particle] * velocity_squared);
            atomic_max(metrics + 1u, sqrt(velocity_squared));
            if (particle % columns != columns - 1u) return;
            atomicAdd(metrics + 4u, static_cast<double>(positions.x[particle]));
            atomicAdd(metrics + 5u, static_cast<double>(positions.y[particle]));
            atomicAdd(metrics + 6u, static_cast<double>(positions.z[particle]));
        }

        __global__ void reduce_strain(const std::uint32_t spring_count, const std::uint32_t* const first, const std::uint32_t* const second, const float* const rest_lengths, const ConstField positions, double* const maximum_absolute_strain) {
            const std::uint32_t spring = blockIdx.x * blockDim.x + threadIdx.x;
            if (spring >= spring_count) return;
            const std::uint32_t first_particle = first[spring];
            const std::uint32_t second_particle = second[spring];
            const double dx = static_cast<double>(positions.x[second_particle]) - positions.x[first_particle];
            const double dy = static_cast<double>(positions.y[second_particle]) - positions.y[first_particle];
            const double dz = static_cast<double>(positions.z[second_particle]) - positions.z[first_particle];
            const double strain = fabs(sqrt(dx * dx + dy * dy + dz * dz) / rest_lengths[spring] - 1.0);
            atomic_max(maximum_absolute_strain, strain);
        }

        std::uint32_t blocks(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }

        void check_launch() {
            if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        }

    } // namespace

    void launch_write_control(const cudaStream_t stream, const std::uint32_t rows, const std::uint32_t columns, const std::uint64_t step, const float time_step, const float mass, const float ramp_duration, const float period, const float base_acceleration, const float primary_acceleration, const float secondary_acceleration, const Field external_forces, double* const metrics) {
        const std::uint32_t particle_count = rows * columns;
        write_control<<<blocks(particle_count), block_size, 0u, stream>>>(particle_count, columns, step, time_step, mass, ramp_duration, period, base_acceleration, primary_acceleration, secondary_acceleration, external_forces, metrics);
        check_launch();
    }

    void launch_particle_metrics(const cudaStream_t stream, const std::uint32_t rows, const std::uint32_t columns, const float* const masses, const ConstField positions, const ConstField velocities, double* const metrics) {
        const std::uint32_t particle_count = rows * columns;
        reduce_particles<<<blocks(particle_count), block_size, 0u, stream>>>(particle_count, columns, masses, positions, velocities, metrics);
        check_launch();
    }

    void launch_strain_metrics(const cudaStream_t stream, const std::uint32_t spring_count, const std::uint32_t* const first, const std::uint32_t* const second, const float* const rest_lengths, const ConstField positions, double* const maximum_absolute_strain) {
        if (spring_count == 0u) return;
        reduce_strain<<<blocks(spring_count), block_size, 0u, stream>>>(spring_count, first, second, rest_lengths, positions, maximum_absolute_strain);
        check_launch();
    }

} // namespace xayah::cloth::examples::forward::simulation_cuda
