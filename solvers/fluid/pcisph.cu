#include "pcisph.h"
#include "cuda_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace xayah::fluid::pcisph::cuda_kernel {

    namespace {

        constexpr std::uint32_t block_size = 256u;

        __host__ std::uint32_t blocks(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }

        void check_launch(const char* name) {
            if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(name);
        }

        __device__ Float3 load(const sph::cuda_kernel::ConstVector field, const std::uint32_t index) {
            return {field.x[index], field.y[index], field.z[index]};
        }

        __device__ void store(const sph::cuda_kernel::Vector field, const std::uint32_t index, const Float3 value) {
            field.x[index] = value.x;
            field.y[index] = value.y;
            field.z[index] = value.z;
        }

        __device__ void collision_mask(const sph::cuda_kernel::Domain domain, const Float3 predicted_position, bool& collision_x, bool& collision_y, bool& collision_z) {
            collision_x = predicted_position.x < domain.minimum_x || predicted_position.x > domain.maximum_x;
            collision_y = predicted_position.y < domain.minimum_y || predicted_position.y > domain.maximum_y;
            collision_z = predicted_position.z < domain.minimum_z || predicted_position.z > domain.maximum_z;
        }

        __global__ void predict_forward_kernel(const std::uint32_t particle_count, const float time_step, const sph::cuda_kernel::Domain domain, const sph::cuda_kernel::ConstVector positions, const sph::cuda_kernel::ConstVector velocities, const sph::cuda_kernel::ConstVector non_pressure_accelerations, const sph::cuda_kernel::ConstVector pressure_accelerations, const sph::cuda_kernel::Vector predicted_positions, const sph::cuda_kernel::Vector predicted_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Float3 velocity = add(load(velocities, particle), scale(add(load(non_pressure_accelerations, particle), load(pressure_accelerations, particle)), time_step));
            Float3 position = add(load(positions, particle), scale(velocity, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, position, collision_x, collision_y, collision_z);
            position.x = fminf(domain.maximum_x, fmaxf(domain.minimum_x, position.x));
            position.y = fminf(domain.maximum_y, fmaxf(domain.minimum_y, position.y));
            position.z = fminf(domain.maximum_z, fmaxf(domain.minimum_z, position.z));
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity = {domain.velocity_x, domain.velocity_y, domain.velocity_z};
            else {
                if (collision_x) velocity.x = domain.velocity_x;
                if (collision_y) velocity.y = domain.velocity_y;
                if (collision_z) velocity.z = domain.velocity_z;
            }
            store(predicted_positions, particle, position);
            store(predicted_velocities, particle, velocity);
        }

        __global__ void predict_jvp_kernel(const std::uint32_t particle_count, const float time_step, const sph::cuda_kernel::Domain domain, const sph::cuda_kernel::ConstVector positions, const sph::cuda_kernel::ConstVector velocities, const sph::cuda_kernel::ConstVector non_pressure_accelerations, const sph::cuda_kernel::ConstVector pressure_accelerations, const sph::cuda_kernel::ConstVector position_tangent, const sph::cuda_kernel::ConstVector velocity_tangent, const sph::cuda_kernel::ConstVector non_pressure_acceleration_tangent, const sph::cuda_kernel::ConstVector pressure_acceleration_tangent, const sph::cuda_kernel::Vector predicted_position_tangent, const sph::cuda_kernel::Vector predicted_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(add(load(non_pressure_accelerations, particle), load(pressure_accelerations, particle)), time_step));
            const Float3 predicted_position = add(load(positions, particle), scale(predicted_velocity, time_step));
            Float3 velocity_dot = add(load(velocity_tangent, particle), scale(add(load(non_pressure_acceleration_tangent, particle), load(pressure_acceleration_tangent, particle)), time_step));
            Float3 position_dot = add(load(position_tangent, particle), scale(velocity_dot, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, predicted_position, collision_x, collision_y, collision_z);
            if (collision_x) position_dot.x = 0.0F;
            if (collision_y) position_dot.y = 0.0F;
            if (collision_z) position_dot.z = 0.0F;
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_dot = {};
            else {
                if (collision_x) velocity_dot.x = 0.0F;
                if (collision_y) velocity_dot.y = 0.0F;
                if (collision_z) velocity_dot.z = 0.0F;
            }
            store(predicted_position_tangent, particle, position_dot);
            store(predicted_velocity_tangent, particle, velocity_dot);
        }

        __global__ void predict_vjp_kernel(const std::uint32_t particle_count, const float time_step, const sph::cuda_kernel::Domain domain, const sph::cuda_kernel::ConstVector positions, const sph::cuda_kernel::ConstVector velocities, const sph::cuda_kernel::ConstVector non_pressure_accelerations, const sph::cuda_kernel::ConstVector pressure_accelerations, const sph::cuda_kernel::ConstVectorAdjoint predicted_position_adjoint, const sph::cuda_kernel::ConstVectorAdjoint predicted_velocity_adjoint, const sph::cuda_kernel::VectorAdjoint position_adjoint, const sph::cuda_kernel::VectorAdjoint velocity_adjoint, const sph::cuda_kernel::VectorAdjoint non_pressure_acceleration_adjoint, const sph::cuda_kernel::VectorAdjoint pressure_acceleration_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(add(load(non_pressure_accelerations, particle), load(pressure_accelerations, particle)), time_step));
            const Float3 predicted_position = add(load(positions, particle), scale(predicted_velocity, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, predicted_position, collision_x, collision_y, collision_z);
            double position_x = collision_x ? 0.0 : predicted_position_adjoint.x[particle];
            double position_y = collision_y ? 0.0 : predicted_position_adjoint.y[particle];
            double position_z = collision_z ? 0.0 : predicted_position_adjoint.z[particle];
            double velocity_x = predicted_velocity_adjoint.x[particle] + time_step * position_x;
            double velocity_y = predicted_velocity_adjoint.y[particle] + time_step * position_y;
            double velocity_z = predicted_velocity_adjoint.z[particle] + time_step * position_z;
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_x = velocity_y = velocity_z = 0.0;
            else {
                if (collision_x) velocity_x = 0.0;
                if (collision_y) velocity_y = 0.0;
                if (collision_z) velocity_z = 0.0;
            }
            position_adjoint.x[particle] += position_x;
            position_adjoint.y[particle] += position_y;
            position_adjoint.z[particle] += position_z;
            velocity_adjoint.x[particle] += velocity_x;
            velocity_adjoint.y[particle] += velocity_y;
            velocity_adjoint.z[particle] += velocity_z;
            const double acceleration_x = time_step * velocity_x;
            const double acceleration_y = time_step * velocity_y;
            const double acceleration_z = time_step * velocity_z;
            non_pressure_acceleration_adjoint.x[particle] += acceleration_x;
            non_pressure_acceleration_adjoint.y[particle] += acceleration_y;
            non_pressure_acceleration_adjoint.z[particle] += acceleration_z;
            pressure_acceleration_adjoint.x[particle] += acceleration_x;
            pressure_acceleration_adjoint.y[particle] += acceleration_y;
            pressure_acceleration_adjoint.z[particle] += acceleration_z;
        }

        __global__ void pressure_update_forward_kernel(const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const sph::cuda_kernel::ParticleParameters particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, float* pressures) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float volume = particles.masses[particle] / particles.rest_densities[particle];
            const float delta = 1.0F / (2.0F * time_step * time_step * volume * volume * reference_gradient_norm);
            pressures[particle] = fmaxf(0.0F, previous_pressures[particle] + pressure_relaxation[particle] * delta * (predicted_densities[particle] - particles.rest_densities[particle]));
        }

        __global__ void pressure_update_jvp_kernel(const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const sph::cuda_kernel::ParticleParameters particles, const sph::cuda_kernel::ParticleParameterTangent particle_tangent, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const float* previous_pressure_tangent, const float* predicted_density_tangent, const float* pressure_relaxation_tangent, float* pressure_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float error = predicted_densities[particle] - particles.rest_densities[particle];
            const float volume = particles.masses[particle] / particles.rest_densities[particle];
            const float delta = 1.0F / (2.0F * time_step * time_step * volume * volume * reference_gradient_norm);
            const float delta_dot = -2.0F * delta * (particle_tangent.masses[particle] / particles.masses[particle] - particle_tangent.rest_densities[particle] / particles.rest_densities[particle]);
            const float candidate = previous_pressures[particle] + pressure_relaxation[particle] * delta * error;
            pressure_tangent[particle] = candidate > 0.0F ? previous_pressure_tangent[particle] + pressure_relaxation_tangent[particle] * delta * error + pressure_relaxation[particle] * delta_dot * error + pressure_relaxation[particle] * delta * (predicted_density_tangent[particle] - particle_tangent.rest_densities[particle]) : 0.0F;
        }

        __global__ void pressure_update_vjp_kernel(const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const sph::cuda_kernel::ParticleParameters particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const double* pressure_adjoint, const sph::cuda_kernel::ParticleParameterAdjoint particle_adjoint, double* previous_pressure_adjoint, double* predicted_density_adjoint, double* pressure_relaxation_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const double error = static_cast<double>(predicted_densities[particle]) - particles.rest_densities[particle];
            const double volume = static_cast<double>(particles.masses[particle]) / particles.rest_densities[particle];
            const double delta = 1.0 / (2.0 * time_step * time_step * volume * volume * reference_gradient_norm);
            const double candidate = previous_pressures[particle] + pressure_relaxation[particle] * delta * error;
            if (candidate > 0.0) {
                const double adjoint = pressure_adjoint[particle];
                previous_pressure_adjoint[particle] += adjoint;
                predicted_density_adjoint[particle] += pressure_relaxation[particle] * delta * adjoint;
                particle_adjoint.rest_densities[particle] += pressure_relaxation[particle] * delta * adjoint * (-1.0 + 2.0 * error / particles.rest_densities[particle]);
                particle_adjoint.masses[particle] -= 2.0 * pressure_relaxation[particle] * delta * error * adjoint / particles.masses[particle];
                pressure_relaxation_adjoint[particle] += delta * error * adjoint;
            }
        }

        __global__ void copy_iteration_forward_kernel(const std::uint32_t particle_count, const float* source_pressures, const float* source_densities, const sph::cuda_kernel::ConstVector source_accelerations, const sph::cuda_kernel::ConstVector source_positions, const sph::cuda_kernel::ConstVector source_velocities, float* destination_pressures, float* destination_densities, const sph::cuda_kernel::Vector destination_accelerations, const sph::cuda_kernel::Vector destination_positions, const sph::cuda_kernel::Vector destination_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            destination_pressures[particle] = source_pressures[particle];
            destination_densities[particle] = source_densities[particle];
            store(destination_accelerations, particle, load(source_accelerations, particle));
            store(destination_positions, particle, load(source_positions, particle));
            store(destination_velocities, particle, load(source_velocities, particle));
        }

    } // namespace

    void launch_predict_forward(void* stream, const std::uint32_t particle_count, const float time_step, const sph::cuda_kernel::Domain domain, const sph::cuda_kernel::ConstVector positions, const sph::cuda_kernel::ConstVector velocities, const sph::cuda_kernel::ConstVector non_pressure_accelerations, const sph::cuda_kernel::ConstVector pressure_accelerations, const sph::cuda_kernel::Vector predicted_positions, const sph::cuda_kernel::Vector predicted_velocities) {
        predict_forward_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, domain, positions, velocities, non_pressure_accelerations, pressure_accelerations, predicted_positions, predicted_velocities);
        check_launch("launch_predict_forward");
    }

    void launch_predict_jvp(void* stream, const std::uint32_t particle_count, const float time_step, const sph::cuda_kernel::Domain domain, const sph::cuda_kernel::ConstVector positions, const sph::cuda_kernel::ConstVector velocities, const sph::cuda_kernel::ConstVector non_pressure_accelerations, const sph::cuda_kernel::ConstVector pressure_accelerations, const sph::cuda_kernel::ConstVector position_tangent, const sph::cuda_kernel::ConstVector velocity_tangent, const sph::cuda_kernel::ConstVector non_pressure_acceleration_tangent, const sph::cuda_kernel::ConstVector pressure_acceleration_tangent, const sph::cuda_kernel::Vector predicted_position_tangent, const sph::cuda_kernel::Vector predicted_velocity_tangent) {
        predict_jvp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, domain, positions, velocities, non_pressure_accelerations, pressure_accelerations, position_tangent, velocity_tangent, non_pressure_acceleration_tangent, pressure_acceleration_tangent, predicted_position_tangent, predicted_velocity_tangent);
        check_launch("launch_predict_jvp");
    }

    void launch_predict_vjp(void* stream, const std::uint32_t particle_count, const float time_step, const sph::cuda_kernel::Domain domain, const sph::cuda_kernel::ConstVector positions, const sph::cuda_kernel::ConstVector velocities, const sph::cuda_kernel::ConstVector non_pressure_accelerations, const sph::cuda_kernel::ConstVector pressure_accelerations, const sph::cuda_kernel::ConstVectorAdjoint predicted_position_adjoint, const sph::cuda_kernel::ConstVectorAdjoint predicted_velocity_adjoint, const sph::cuda_kernel::VectorAdjoint position_adjoint, const sph::cuda_kernel::VectorAdjoint velocity_adjoint, const sph::cuda_kernel::VectorAdjoint non_pressure_acceleration_adjoint, const sph::cuda_kernel::VectorAdjoint pressure_acceleration_adjoint) {
        predict_vjp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, domain, positions, velocities, non_pressure_accelerations, pressure_accelerations, predicted_position_adjoint, predicted_velocity_adjoint, position_adjoint, velocity_adjoint, non_pressure_acceleration_adjoint, pressure_acceleration_adjoint);
        check_launch("launch_predict_vjp");
    }

    void launch_pressure_update_forward(void* stream, const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const sph::cuda_kernel::ParticleParameters particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, float* pressures) {
        pressure_update_forward_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, reference_gradient_norm, particles, previous_pressures, predicted_densities, pressure_relaxation, pressures);
        check_launch("launch_pressure_update_forward");
    }

    void launch_pressure_update_jvp(void* stream, const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const sph::cuda_kernel::ParticleParameters particles, const sph::cuda_kernel::ParticleParameterTangent particle_tangent, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const float* previous_pressure_tangent, const float* predicted_density_tangent, const float* pressure_relaxation_tangent, float* pressure_tangent) {
        pressure_update_jvp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, reference_gradient_norm, particles, particle_tangent, previous_pressures, predicted_densities, pressure_relaxation, previous_pressure_tangent, predicted_density_tangent, pressure_relaxation_tangent, pressure_tangent);
        check_launch("launch_pressure_update_jvp");
    }

    void launch_pressure_update_vjp(void* stream, const std::uint32_t particle_count, const float time_step, const float reference_gradient_norm, const sph::cuda_kernel::ParticleParameters particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const double* pressure_adjoint, const sph::cuda_kernel::ParticleParameterAdjoint particle_adjoint, double* previous_pressure_adjoint, double* predicted_density_adjoint, double* pressure_relaxation_adjoint) {
        pressure_update_vjp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, reference_gradient_norm, particles, previous_pressures, predicted_densities, pressure_relaxation, pressure_adjoint, particle_adjoint, previous_pressure_adjoint, predicted_density_adjoint, pressure_relaxation_adjoint);
        check_launch("launch_pressure_update_vjp");
    }

    void launch_copy_iteration_forward(void* stream, const std::uint32_t particle_count, const float* source_pressures, const float* source_densities, const sph::cuda_kernel::ConstVector source_accelerations, const sph::cuda_kernel::ConstVector source_positions, const sph::cuda_kernel::ConstVector source_velocities, float* destination_pressures, float* destination_densities, const sph::cuda_kernel::Vector destination_accelerations, const sph::cuda_kernel::Vector destination_positions, const sph::cuda_kernel::Vector destination_velocities) {
        copy_iteration_forward_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, source_pressures, source_densities, source_accelerations, source_positions, source_velocities, destination_pressures, destination_densities, destination_accelerations, destination_positions, destination_velocities);
        check_launch("launch_copy_iteration_forward");
    }

} // namespace xayah::fluid::pcisph::cuda_kernel
