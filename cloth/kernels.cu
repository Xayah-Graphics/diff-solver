#include "kernels.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace xayah::cloth::cuda_kernel {

    namespace {

        constexpr std::uint32_t block_size = 256;

        struct Vector {
            float x;
            float y;
            float z;
        };

        __device__ Vector operator+(const Vector left, const Vector right) {
            return {left.x + right.x, left.y + right.y, left.z + right.z};
        }

        __device__ Vector operator-(const Vector left, const Vector right) {
            return {left.x - right.x, left.y - right.y, left.z - right.z};
        }

        __device__ Vector operator*(const float scalar, const Vector vector) {
            return {scalar * vector.x, scalar * vector.y, scalar * vector.z};
        }

        __device__ Vector operator/(const Vector vector, const float scalar) {
            return {vector.x / scalar, vector.y / scalar, vector.z / scalar};
        }

        __device__ float dot(const Vector left, const Vector right) {
            return left.x * right.x + left.y * right.y + left.z * right.z;
        }

        __device__ Vector load(const ConstField field, const std::uint32_t index) {
            return {field.x[index], field.y[index], field.z[index]};
        }

        __device__ void store(const Field field, const std::uint32_t index, const Vector value) {
            field.x[index] = value.x;
            field.y[index] = value.y;
            field.z[index] = value.z;
        }

        __device__ void add(const Field field, const std::uint32_t index, const Vector value) {
            field.x[index] += value.x;
            field.y[index] += value.y;
            field.z[index] += value.z;
        }

        __device__ Vector spring_force(const Vector first_position, const Vector second_position, const Vector first_velocity, const Vector second_velocity, const float stiffness, const float damping, const float rest_length) {
            const Vector displacement      = second_position - first_position;
            const float length             = sqrtf(dot(displacement, displacement));
            const Vector direction         = displacement / length;
            const Vector relative_velocity = second_velocity - first_velocity;
            const float magnitude          = stiffness * (length - rest_length) + damping * dot(relative_velocity, direction);
            return magnitude * direction;
        }

        __device__ Vector spring_force_tangent(const Vector first_position, const Vector second_position, const Vector first_velocity, const Vector second_velocity, const Vector first_position_tangent, const Vector second_position_tangent, const Vector first_velocity_tangent, const Vector second_velocity_tangent, const float stiffness, const float damping, const float rest_length, const float stiffness_tangent, const float damping_tangent, const float rest_length_tangent) {
            const Vector displacement              = second_position - first_position;
            const Vector displacement_tangent      = second_position_tangent - first_position_tangent;
            const float length                     = sqrtf(dot(displacement, displacement));
            const Vector direction                 = displacement / length;
            const float length_tangent             = dot(direction, displacement_tangent);
            const Vector direction_tangent         = (displacement_tangent - length_tangent * direction) / length;
            const Vector relative_velocity         = second_velocity - first_velocity;
            const Vector relative_velocity_tangent = second_velocity_tangent - first_velocity_tangent;
            const float axial_velocity             = dot(relative_velocity, direction);
            const float axial_velocity_tangent     = dot(relative_velocity_tangent, direction) + dot(relative_velocity, direction_tangent);
            const float magnitude                  = stiffness * (length - rest_length) + damping * axial_velocity;
            const float magnitude_tangent          = stiffness_tangent * (length - rest_length) + stiffness * (length_tangent - rest_length_tangent) + damping_tangent * axial_velocity + damping * axial_velocity_tangent;
            return magnitude_tangent * direction + magnitude * direction_tangent;
        }

        __device__ void spring_vjp(const Vector first_position, const Vector second_position, const Vector first_velocity, const Vector second_velocity, const float stiffness, const float damping, const float rest_length, const Vector force_adjoint, Vector& displacement_adjoint, Vector& relative_velocity_adjoint, float& stiffness_adjoint, float& damping_adjoint, float& rest_length_adjoint) {
            const Vector displacement      = second_position - first_position;
            const float length             = sqrtf(dot(displacement, displacement));
            const Vector direction         = displacement / length;
            const Vector relative_velocity = second_velocity - first_velocity;
            const float axial_velocity     = dot(relative_velocity, direction);
            const float magnitude          = stiffness * (length - rest_length) + damping * axial_velocity;
            const float magnitude_adjoint  = dot(force_adjoint, direction);
            Vector direction_adjoint       = magnitude * force_adjoint;
            const float axial_adjoint      = damping * magnitude_adjoint;
            const float length_adjoint     = stiffness * magnitude_adjoint;
            stiffness_adjoint              = (length - rest_length) * magnitude_adjoint;
            damping_adjoint                = axial_velocity * magnitude_adjoint;
            rest_length_adjoint            = -stiffness * magnitude_adjoint;
            relative_velocity_adjoint      = axial_adjoint * direction;
            direction_adjoint              = direction_adjoint + axial_adjoint * relative_velocity;
            displacement_adjoint           = length_adjoint * direction + (direction_adjoint - dot(direction, direction_adjoint) * direction) / length;
        }

        __device__ Vector gathered_force(const std::uint32_t particle, const ConstField positions, const ConstField velocities, const SpringTopology topology, const SpringParameters parameters) {
            Vector result{0.0F, 0.0F, 0.0F};
            for (std::uint32_t entry = topology.offsets[particle]; entry < topology.offsets[particle + 1]; ++entry) {
                const std::uint32_t spring   = topology.indices[entry];
                const std::uint32_t other    = topology.others[entry];
                const float sign             = topology.signs[entry];
                const Vector first_position  = sign > 0.0F ? load(positions, particle) : load(positions, other);
                const Vector second_position = sign > 0.0F ? load(positions, other) : load(positions, particle);
                const Vector first_velocity  = sign > 0.0F ? load(velocities, particle) : load(velocities, other);
                const Vector second_velocity = sign > 0.0F ? load(velocities, other) : load(velocities, particle);
                result                       = result + sign * spring_force(first_position, second_position, first_velocity, second_velocity, parameters.stiffnesses[spring], parameters.dampings[spring], parameters.rest_lengths[spring]);
            }
            return result;
        }

        __device__ Vector gathered_force_tangent(const std::uint32_t particle, const ConstField positions, const ConstField velocities, const ConstField positions_tangent, const ConstField velocities_tangent, const SpringTopology topology, const SpringParameters parameters, const SpringParameterTangents tangents) {
            Vector result{0.0F, 0.0F, 0.0F};
            for (std::uint32_t entry = topology.offsets[particle]; entry < topology.offsets[particle + 1]; ++entry) {
                const std::uint32_t spring           = topology.indices[entry];
                const std::uint32_t other            = topology.others[entry];
                const float sign                     = topology.signs[entry];
                const Vector first_position          = sign > 0.0F ? load(positions, particle) : load(positions, other);
                const Vector second_position         = sign > 0.0F ? load(positions, other) : load(positions, particle);
                const Vector first_velocity          = sign > 0.0F ? load(velocities, particle) : load(velocities, other);
                const Vector second_velocity         = sign > 0.0F ? load(velocities, other) : load(velocities, particle);
                const Vector first_position_tangent  = sign > 0.0F ? load(positions_tangent, particle) : load(positions_tangent, other);
                const Vector second_position_tangent = sign > 0.0F ? load(positions_tangent, other) : load(positions_tangent, particle);
                const Vector first_velocity_tangent  = sign > 0.0F ? load(velocities_tangent, particle) : load(velocities_tangent, other);
                const Vector second_velocity_tangent = sign > 0.0F ? load(velocities_tangent, other) : load(velocities_tangent, particle);
                result                               = result + sign * spring_force_tangent(first_position, second_position, first_velocity, second_velocity, first_position_tangent, second_position_tangent, first_velocity_tangent, second_velocity_tangent, parameters.stiffnesses[spring], parameters.dampings[spring], parameters.rest_lengths[spring], tangents.stiffnesses[spring], tangents.dampings[spring], tangents.rest_lengths[spring]);
            }
            return result;
        }

        __device__ void gathered_state_vjp(const std::uint32_t particle, const ConstField positions, const ConstField velocities, const ConstField force_adjoint, const SpringTopology topology, const SpringParameters parameters, Vector& position_adjoint, Vector& velocity_adjoint) {
            for (std::uint32_t entry = topology.offsets[particle]; entry < topology.offsets[particle + 1]; ++entry) {
                const std::uint32_t spring       = topology.indices[entry];
                const float sign                 = topology.signs[entry];
                const std::uint32_t first        = topology.first[spring];
                const std::uint32_t second       = topology.second[spring];
                const Vector local_force_adjoint = load(force_adjoint, first) - load(force_adjoint, second);
                Vector displacement_adjoint;
                Vector relative_velocity_adjoint;
                float stiffness_adjoint;
                float damping_adjoint;
                float rest_length_adjoint;
                spring_vjp(load(positions, first), load(positions, second), load(velocities, first), load(velocities, second), parameters.stiffnesses[spring], parameters.dampings[spring], parameters.rest_lengths[spring], local_force_adjoint, displacement_adjoint, relative_velocity_adjoint, stiffness_adjoint, damping_adjoint, rest_length_adjoint);
                position_adjoint = position_adjoint - sign * displacement_adjoint;
                velocity_adjoint = velocity_adjoint - sign * relative_velocity_adjoint;
            }
        }

        __global__ void force_forward_kernel(const std::uint32_t particle_count, const Vector gravity, const ConstField positions, const ConstField velocities, const ConstField controls, const float* masses, const SpringTopology stretch_topology, const SpringParameters stretch_parameters, const SpringTopology bending_topology, const SpringParameters bending_parameters, const Field forces) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector force = load(controls, particle) + masses[particle] * gravity + gathered_force(particle, positions, velocities, stretch_topology, stretch_parameters) + gathered_force(particle, positions, velocities, bending_topology, bending_parameters);
            store(forces, particle, force);
        }

        __global__ void force_jvp_kernel(const std::uint32_t particle_count, const Vector gravity, const ConstField positions, const ConstField velocities, const ConstField controls_tangent, const ConstField positions_tangent, const ConstField velocities_tangent, const float* masses_tangent, const SpringTopology stretch_topology, const SpringParameters stretch_parameters, const SpringParameterTangents stretch_tangents, const SpringTopology bending_topology, const SpringParameters bending_parameters, const SpringParameterTangents bending_tangents, const Field force_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector tangent = load(controls_tangent, particle) + masses_tangent[particle] * gravity + gathered_force_tangent(particle, positions, velocities, positions_tangent, velocities_tangent, stretch_topology, stretch_parameters, stretch_tangents) + gathered_force_tangent(particle, positions, velocities, positions_tangent, velocities_tangent, bending_topology, bending_parameters, bending_tangents);
            store(force_tangent, particle, tangent);
        }

        __global__ void force_state_vjp_kernel(const std::uint32_t particle_count, const Vector gravity, const ConstField positions, const ConstField velocities, const ConstField force_adjoint, const SpringTopology stretch_topology, const SpringParameters stretch_parameters, const SpringTopology bending_topology, const SpringParameters bending_parameters, const Field state_position_adjoint, const Field state_velocity_adjoint, const Field control_adjoint, float* mass_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Vector position_adjoint{0.0F, 0.0F, 0.0F};
            Vector velocity_adjoint{0.0F, 0.0F, 0.0F};
            gathered_state_vjp(particle, positions, velocities, force_adjoint, stretch_topology, stretch_parameters, position_adjoint, velocity_adjoint);
            gathered_state_vjp(particle, positions, velocities, force_adjoint, bending_topology, bending_parameters, position_adjoint, velocity_adjoint);
            add(state_position_adjoint, particle, position_adjoint);
            add(state_velocity_adjoint, particle, velocity_adjoint);
            const Vector local_force_adjoint = load(force_adjoint, particle);
            add(control_adjoint, particle, local_force_adjoint);
            mass_adjoint[particle] += dot(gravity, local_force_adjoint);
        }

        __global__ void force_parameter_vjp_kernel(const std::uint32_t spring_count, const ConstField positions, const ConstField velocities, const ConstField force_adjoint, const SpringTopology topology, const SpringParameters parameters, const SpringParameterAdjoints parameter_adjoint) {
            const std::uint32_t spring = blockIdx.x * blockDim.x + threadIdx.x;
            if (spring >= spring_count) return;
            const std::uint32_t first  = topology.first[spring];
            const std::uint32_t second = topology.second[spring];
            Vector displacement_adjoint;
            Vector relative_velocity_adjoint;
            float stiffness_adjoint;
            float damping_adjoint;
            float rest_length_adjoint;
            spring_vjp(load(positions, first), load(positions, second), load(velocities, first), load(velocities, second), parameters.stiffnesses[spring], parameters.dampings[spring], parameters.rest_lengths[spring], load(force_adjoint, first) - load(force_adjoint, second), displacement_adjoint, relative_velocity_adjoint, stiffness_adjoint, damping_adjoint, rest_length_adjoint);
            parameter_adjoint.stiffnesses[spring] += stiffness_adjoint;
            parameter_adjoint.dampings[spring] += damping_adjoint;
            parameter_adjoint.rest_lengths[spring] += rest_length_adjoint;
        }

        __global__ void euler_forward_kernel(const std::uint32_t particle_count, const float time_step, const ConstField positions, const ConstField velocities, const ConstField forces, const float* masses, const Field integrated_positions, const Field integrated_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector velocity = load(velocities, particle) + (time_step / masses[particle]) * load(forces, particle);
            store(integrated_velocities, particle, velocity);
            store(integrated_positions, particle, load(positions, particle) + time_step * velocity);
        }

        __global__ void euler_jvp_kernel(const std::uint32_t particle_count, const float time_step, const ConstField forces, const float* masses, const ConstField position_tangent, const ConstField velocity_tangent, const ConstField force_tangent, const float* mass_tangent, const Field integrated_position_tangent, const Field integrated_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const float mass      = masses[particle];
            const Vector velocity = load(velocity_tangent, particle) + time_step * (load(force_tangent, particle) / mass - (mass_tangent[particle] / (mass * mass)) * load(forces, particle));
            store(integrated_velocity_tangent, particle, velocity);
            store(integrated_position_tangent, particle, load(position_tangent, particle) + time_step * velocity);
        }

        __global__ void euler_vjp_kernel(const std::uint32_t particle_count, const float time_step, const ConstField forces, const float* masses, const ConstField integrated_position_adjoint, const ConstField integrated_velocity_adjoint, const Field state_position_adjoint, const Field state_velocity_adjoint, const Field force_adjoint, float* mass_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position_adjoint = load(integrated_position_adjoint, particle);
            const Vector velocity_adjoint = load(integrated_velocity_adjoint, particle) + time_step * position_adjoint;
            add(state_position_adjoint, particle, position_adjoint);
            add(state_velocity_adjoint, particle, velocity_adjoint);
            add(force_adjoint, particle, (time_step / masses[particle]) * velocity_adjoint);
            mass_adjoint[particle] -= time_step * dot(load(forces, particle), velocity_adjoint) / (masses[particle] * masses[particle]);
        }

        __global__ void constraint_forward_kernel(const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const ConstField anchor_positions, const ConstField positions, const ConstField velocities, const Field constrained_positions, const Field constrained_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            if (anchor_mask[particle] != 0) {
                store(constrained_positions, particle, load(anchor_positions, particle));
                store(constrained_velocities, particle, {0.0F, 0.0F, 0.0F});
            } else {
                store(constrained_positions, particle, load(positions, particle));
                store(constrained_velocities, particle, load(velocities, particle));
            }
        }

        __global__ void constraint_jvp_kernel(const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const ConstField position_tangent, const ConstField velocity_tangent, const Field constrained_position_tangent, const Field constrained_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            if (anchor_mask[particle] != 0) {
                store(constrained_position_tangent, particle, {0.0F, 0.0F, 0.0F});
                store(constrained_velocity_tangent, particle, {0.0F, 0.0F, 0.0F});
            } else {
                store(constrained_position_tangent, particle, load(position_tangent, particle));
                store(constrained_velocity_tangent, particle, load(velocity_tangent, particle));
            }
        }

        __global__ void constraint_vjp_kernel(const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const ConstField constrained_position_adjoint, const ConstField constrained_velocity_adjoint, const Field position_adjoint, const Field velocity_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count || anchor_mask[particle] != 0) return;
            add(position_adjoint, particle, load(constrained_position_adjoint, particle));
            add(velocity_adjoint, particle, load(constrained_velocity_adjoint, particle));
        }

        __global__ void accumulate_kernel(const std::uint32_t count, const ConstField source, const Field destination) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            add(destination, index, load(source, index));
        }

        void check_launch() {
            const cudaError_t result = cudaGetLastError();
            if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        }

        std::uint32_t grid_size(const std::uint32_t count) {
            return (count + block_size - 1) / block_size;
        }

    } // namespace

    void launch_force_forward(void* stream, const std::uint32_t particle_count, const float gravity_x, const float gravity_y, const float gravity_z, const ConstField positions, const ConstField velocities, const ConstField controls, const float* masses, const SpringTopology stretch_topology, const SpringParameters stretch_parameters, const SpringTopology bending_topology, const SpringParameters bending_parameters, const Field forces) {
        force_forward_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, {gravity_x, gravity_y, gravity_z}, positions, velocities, controls, masses, stretch_topology, stretch_parameters, bending_topology, bending_parameters, forces);
        check_launch();
    }

    void launch_force_jvp(void* stream, const std::uint32_t particle_count, const float gravity_x, const float gravity_y, const float gravity_z, const ConstField positions, const ConstField velocities, const ConstField controls_tangent, const ConstField positions_tangent, const ConstField velocities_tangent, const float* masses_tangent, const SpringTopology stretch_topology, const SpringParameters stretch_parameters, const SpringParameterTangents stretch_tangents, const SpringTopology bending_topology, const SpringParameters bending_parameters, const SpringParameterTangents bending_tangents, const Field force_tangent) {
        force_jvp_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, {gravity_x, gravity_y, gravity_z}, positions, velocities, controls_tangent, positions_tangent, velocities_tangent, masses_tangent, stretch_topology, stretch_parameters, stretch_tangents, bending_topology, bending_parameters, bending_tangents, force_tangent);
        check_launch();
    }

    void launch_force_state_vjp(void* stream, const std::uint32_t particle_count, const float gravity_x, const float gravity_y, const float gravity_z, const ConstField positions, const ConstField velocities, const ConstField force_adjoint, const SpringTopology stretch_topology, const SpringParameters stretch_parameters, const SpringTopology bending_topology, const SpringParameters bending_parameters, const Field state_position_adjoint, const Field state_velocity_adjoint, const Field control_adjoint, float* mass_adjoint) {
        force_state_vjp_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, {gravity_x, gravity_y, gravity_z}, positions, velocities, force_adjoint, stretch_topology, stretch_parameters, bending_topology, bending_parameters, state_position_adjoint, state_velocity_adjoint, control_adjoint, mass_adjoint);
        check_launch();
    }

    void launch_force_parameter_vjp(void* stream, const std::uint32_t spring_count, const ConstField positions, const ConstField velocities, const ConstField force_adjoint, const SpringTopology topology, const SpringParameters parameters, const SpringParameterAdjoints parameter_adjoint) {
        if (spring_count == 0) return;
        force_parameter_vjp_kernel<<<grid_size(spring_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(spring_count, positions, velocities, force_adjoint, topology, parameters, parameter_adjoint);
        check_launch();
    }

    void launch_euler_forward(void* stream, const std::uint32_t particle_count, const float time_step, const ConstField positions, const ConstField velocities, const ConstField forces, const float* masses, const Field integrated_positions, const Field integrated_velocities) {
        euler_forward_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, positions, velocities, forces, masses, integrated_positions, integrated_velocities);
        check_launch();
    }

    void launch_euler_jvp(void* stream, const std::uint32_t particle_count, const float time_step, const ConstField forces, const float* masses, const ConstField position_tangent, const ConstField velocity_tangent, const ConstField force_tangent, const float* mass_tangent, const Field integrated_position_tangent, const Field integrated_velocity_tangent) {
        euler_jvp_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, forces, masses, position_tangent, velocity_tangent, force_tangent, mass_tangent, integrated_position_tangent, integrated_velocity_tangent);
        check_launch();
    }

    void launch_euler_vjp(void* stream, const std::uint32_t particle_count, const float time_step, const ConstField forces, const float* masses, const ConstField integrated_position_adjoint, const ConstField integrated_velocity_adjoint, const Field state_position_adjoint, const Field state_velocity_adjoint, const Field force_adjoint, float* mass_adjoint) {
        euler_vjp_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, forces, masses, integrated_position_adjoint, integrated_velocity_adjoint, state_position_adjoint, state_velocity_adjoint, force_adjoint, mass_adjoint);
        check_launch();
    }

    void launch_constraint_forward(void* stream, const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const ConstField anchor_positions, const ConstField positions, const ConstField velocities, const Field constrained_positions, const Field constrained_velocities) {
        constraint_forward_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, anchor_mask, anchor_positions, positions, velocities, constrained_positions, constrained_velocities);
        check_launch();
    }

    void launch_constraint_jvp(void* stream, const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const ConstField position_tangent, const ConstField velocity_tangent, const Field constrained_position_tangent, const Field constrained_velocity_tangent) {
        constraint_jvp_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, anchor_mask, position_tangent, velocity_tangent, constrained_position_tangent, constrained_velocity_tangent);
        check_launch();
    }

    void launch_constraint_vjp(void* stream, const std::uint32_t particle_count, const std::uint32_t* anchor_mask, const ConstField constrained_position_adjoint, const ConstField constrained_velocity_adjoint, const Field position_adjoint, const Field velocity_adjoint) {
        constraint_vjp_kernel<<<grid_size(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, anchor_mask, constrained_position_adjoint, constrained_velocity_adjoint, position_adjoint, velocity_adjoint);
        check_launch();
    }

    void launch_accumulate(void* stream, const std::uint32_t count, const ConstField source, const Field destination) {
        accumulate_kernel<<<grid_size(count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(count, source, destination);
        check_launch();
    }

} // namespace xayah::cloth::cuda_kernel
