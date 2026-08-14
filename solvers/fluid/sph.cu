#include "sph.h"
#include "cuda_common.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace xayah::fluid::sph::cuda_kernel {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        constexpr float pi = 3.14159265358979323846F;

        __host__ std::uint32_t blocks(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }

        void check_launch(const char* name) {
            if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(name);
        }

        __device__ Float3 load(const ConstVector field, const std::uint32_t index) {
            return {field.x[index], field.y[index], field.z[index]};
        }

        __device__ Float3 load_boundary_position(const Boundary boundary, const std::uint32_t index) {
            return {
                boundary.position_x[index] + boundary.time * boundary.velocity_x[index],
                boundary.position_y[index] + boundary.time * boundary.velocity_y[index],
                boundary.position_z[index] + boundary.time * boundary.velocity_z[index],
            };
        }

        __device__ Float3 load_boundary_velocity(const Boundary boundary, const std::uint32_t index) {
            return {boundary.velocity_x[index], boundary.velocity_y[index], boundary.velocity_z[index]};
        }

        __device__ Double3 load(const ConstVectorAdjoint field, const std::uint32_t index) {
            return {field.x[index], field.y[index], field.z[index]};
        }

        __device__ void accumulate(const VectorAdjoint field, const std::uint32_t index, const Double3 value) {
            field.x[index] += value.x;
            field.y[index] += value.y;
            field.z[index] += value.z;
        }

        __device__ void store(const Vector field, const std::uint32_t index, const Float3 value) {
            field.x[index] = value.x;
            field.y[index] = value.y;
            field.z[index] = value.z;
        }

        __device__ std::uint32_t lower_bound(const std::uint64_t* keys, const std::uint32_t count, const std::uint64_t key) {
            std::uint32_t first = 0u;
            std::uint32_t last = count;
            while (first < last) {
                const std::uint32_t middle = first + (last - first) / 2u;
                if (keys[middle] < key) first = middle + 1u;
                else last = middle;
            }
            return first;
        }

        __device__ float poly6(const Float3 displacement, const float support_radius) {
            const float squared_distance = dot(displacement, displacement);
            const float squared_radius = support_radius * support_radius;
            if (squared_distance >= squared_radius) return 0.0F;
            const float difference = squared_radius - squared_distance;
            const float radius_squared = squared_radius;
            const float coefficient = 315.0F / (64.0F * pi * radius_squared * radius_squared * radius_squared * support_radius * support_radius * support_radius);
            return coefficient * difference * difference * difference;
        }

        __device__ Float3 poly6_gradient(const Float3 displacement, const float support_radius) {
            const float squared_distance = dot(displacement, displacement);
            const float squared_radius = support_radius * support_radius;
            if (squared_distance >= squared_radius) return {};
            const float difference = squared_radius - squared_distance;
            const float coefficient = -945.0F / (32.0F * pi * squared_radius * squared_radius * squared_radius * support_radius * support_radius * support_radius);
            return scale(displacement, coefficient * difference * difference);
        }

        __device__ float cubic(const Float3 displacement, const float support_radius) {
            const float distance = length(displacement);
            if (distance >= support_radius) return 0.0F;
            const float q = distance / support_radius;
            const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
            if (q <= 0.5F) return coefficient * (6.0F * q * q * q - 6.0F * q * q + 1.0F);
            const float difference = 1.0F - q;
            return coefficient * 2.0F * difference * difference * difference;
        }

        __device__ Float3 cubic_gradient(const Float3 displacement, const float support_radius) {
            const float distance = length(displacement);
            if (distance == 0.0F || distance >= support_radius) return {};
            const float q = distance / support_radius;
            const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
            const float derivative = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
            return scale(displacement, coefficient * derivative / (support_radius * distance));
        }

        __device__ Float3 cubic_gradient_tangent(const Float3 displacement, const Float3 displacement_tangent, const float support_radius) {
            const float distance = length(displacement);
            const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
            if (distance == 0.0F) return scale(displacement_tangent, -12.0F * coefficient / (support_radius * support_radius));
            if (distance >= support_radius) return {};
            const float q = distance / support_radius;
            const float derivative = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
            const float second_derivative = q <= 0.5F ? 36.0F * q - 12.0F : 12.0F * (1.0F - q);
            const float radial_tangent = dot(displacement, displacement_tangent) / distance;
            const float radial_scale = coefficient * derivative / (support_radius * distance);
            const float radial_scale_derivative = coefficient * (second_derivative / (support_radius * support_radius * distance) - derivative / (support_radius * distance * distance));
            return add(scale(displacement_tangent, radial_scale), scale(displacement, radial_scale_derivative * radial_tangent));
        }

        __device__ Double3 cubic_hessian_product(const Float3 displacement, const Double3 vector, const float support_radius) {
            const double distance = sqrt(static_cast<double>(dot(displacement, displacement)));
            const double coefficient = 8.0 / (static_cast<double>(pi) * support_radius * support_radius * support_radius);
            if (distance == 0.0) return scale(vector, -12.0 * coefficient / (support_radius * support_radius));
            if (distance >= support_radius) return {};
            const double q = distance / support_radius;
            const double derivative = q <= 0.5 ? 18.0 * q * q - 12.0 * q : -6.0 * (1.0 - q) * (1.0 - q);
            const double second_derivative = q <= 0.5 ? 36.0 * q - 12.0 : 12.0 * (1.0 - q);
            const double radial_scale = coefficient * derivative / (support_radius * distance);
            const double radial_scale_derivative = coefficient * (second_derivative / (support_radius * support_radius * distance) - derivative / (support_radius * distance * distance));
            const double projection = dot(vector, displacement) / distance;
            return add(scale(vector, radial_scale), scale(displacement, radial_scale_derivative * projection));
        }

        __device__ float density_kernel(const Float3 displacement, const float support_radius, const std::uint32_t pbf_kernel) {
            return pbf_kernel != 0u ? poly6(displacement, support_radius) : cubic(displacement, support_radius);
        }

        __device__ Float3 density_kernel_gradient(const Float3 displacement, const float support_radius, const std::uint32_t pbf_kernel) {
            return pbf_kernel != 0u ? poly6_gradient(displacement, support_radius) : cubic_gradient(displacement, support_radius);
        }

        __device__ float viscosity_laplacian(const Float3 displacement, const float support_radius) {
            const float distance = length(displacement);
            if (distance >= support_radius) return 0.0F;
            return 45.0F * (support_radius - distance) / (pi * support_radius * support_radius * support_radius * support_radius * support_radius * support_radius);
        }

        __device__ float viscosity_laplacian_tangent(const Float3 displacement, const Float3 displacement_tangent, const float support_radius) {
            const float distance = length(displacement);
            if (distance == 0.0F || distance >= support_radius) return 0.0F;
            const float coefficient = -45.0F / (pi * support_radius * support_radius * support_radius * support_radius * support_radius * support_radius);
            return coefficient * dot(displacement, displacement_tangent) / distance;
        }

        __device__ Double3 viscosity_laplacian_gradient(const Float3 displacement, const float support_radius, const double scalar) {
            const double distance = sqrt(static_cast<double>(dot(displacement, displacement)));
            if (distance == 0.0 || distance >= support_radius) return {};
            const double coefficient = -45.0 / (static_cast<double>(pi) * pow(static_cast<double>(support_radius), 6.0) * distance);
            return scale(displacement, coefficient * scalar);
        }

        __global__ void copy_vector_kernel(const std::uint32_t count, const ConstVector source, const Vector destination) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            destination.x[index] = source.x[index];
            destination.y[index] = source.y[index];
            destination.z[index] = source.z[index];
        }

        __global__ void copy_vector_adjoint_kernel(const std::uint32_t count, const ConstVectorAdjoint source, const VectorAdjoint destination) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            destination.x[index] = source.x[index];
            destination.y[index] = source.y[index];
            destination.z[index] = source.z[index];
        }

        __global__ void accumulate_vector_adjoint_kernel(const std::uint32_t count, const ConstVectorAdjoint source, const VectorAdjoint destination) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            destination.x[index] += source.x[index];
            destination.y[index] += source.y[index];
            destination.z[index] += source.z[index];
        }

        template <typename Value>
        __global__ void copy_scalar_kernel(const std::uint32_t count, const Value* source, Value* destination) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            destination[index] = source[index];
        }

        __global__ void accumulate_scalar_adjoint_kernel(const std::uint32_t count, const double* source, double* destination) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            destination[index] += source[index];
        }

        __global__ void key_kernel(const std::uint32_t count, const float support_radius, const Domain domain, const float time, const std::uint32_t moving_positions, const ConstVector positions, const ConstVector velocities, std::uint64_t* keys, std::uint32_t* indices) {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) return;
            int x, y, z;
            Float3 position = load(positions, index);
            if (moving_positions != 0u) position = add(position, scale(load(velocities, index), time));
            const std::uint32_t cells_x = static_cast<std::uint32_t>(ceilf((domain.maximum_x - domain.minimum_x) / support_radius));
            const std::uint32_t cells_y = static_cast<std::uint32_t>(ceilf((domain.maximum_y - domain.minimum_y) / support_radius));
            const std::uint32_t cells_z = static_cast<std::uint32_t>(ceilf((domain.maximum_z - domain.minimum_z) / support_radius));
            x = min(static_cast<int>(cells_x) - 1, max(0, __float2int_rd((position.x - domain.minimum_x - time * domain.velocity_x) / support_radius)));
            y = min(static_cast<int>(cells_y) - 1, max(0, __float2int_rd((position.y - domain.minimum_y - time * domain.velocity_y) / support_radius)));
            z = min(static_cast<int>(cells_z) - 1, max(0, __float2int_rd((position.z - domain.minimum_z - time * domain.velocity_z) / support_radius)));
            keys[index] = (static_cast<std::uint64_t>(z) * cells_y + static_cast<std::uint64_t>(y)) * cells_x + static_cast<std::uint64_t>(x);
            indices[index] = index;
        }

        __global__ void cell_offsets_kernel(const std::uint32_t item_count, const std::uint64_t* sorted_keys, const std::uint32_t cell_count, std::uint32_t* cell_offsets) {
            const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
            if (cell > cell_count) return;
            cell_offsets[cell] = lower_bound(sorted_keys, item_count, cell);
        }

        __global__ void density_forward_kernel(const std::uint32_t particle_count, const float support_radius, const std::uint32_t pbf_kernel, const ConstVector topology_positions, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, float* densities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            float density = 0.0F;
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            density += parameters.masses[neighbor] * density_kernel(subtract(position, load(positions, neighbor)), support_radius, pbf_kernel);
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            density += parameters.rest_densities[particle] * boundary.volumes[neighbor] * density_kernel(subtract(position, load_boundary_position(boundary, neighbor)), support_radius, pbf_kernel);
                        }
                    }
            densities[particle] = density;
        }

        __global__ void density_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const std::uint32_t pbf_kernel, const ConstVector topology_positions, const ConstVector positions, const ConstVector position_tangent, const ParticleParameters parameters, const ParticleParameterTangent parameter_tangent, const Neighborhood neighborhood, const Boundary boundary, float* density_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 tangent = load(position_tangent, particle);
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            float result = 0.0F;
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            result += parameter_tangent.masses[neighbor] * density_kernel(displacement, support_radius, pbf_kernel) + parameters.masses[neighbor] * dot(density_kernel_gradient(displacement, support_radius, pbf_kernel), subtract(tangent, load(position_tangent, neighbor)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, load_boundary_position(boundary, neighbor));
                            result += boundary.volumes[neighbor] * (parameter_tangent.rest_densities[particle] * density_kernel(displacement, support_radius, pbf_kernel) + parameters.rest_densities[particle] * dot(density_kernel_gradient(displacement, support_radius, pbf_kernel), tangent));
                        }
                    }
            density_tangent[particle] = result;
        }

        __global__ void density_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const std::uint32_t pbf_kernel, const ConstVector topology_positions, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const double* density_adjoint, const VectorAdjoint position_adjoint, const ParticleParameterAdjoint parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const double output_adjoint = density_adjoint[particle];
            Double3 position_contribution{};
            double mass_contribution = 0.0;
            double rest_density_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, load(topology_positions, particle), cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t other = neighborhood.sorted_particle_indices[sorted];
                            const Float3 displacement = subtract(position, load(positions, other));
                            const Float3 gradient = density_kernel_gradient(displacement, support_radius, pbf_kernel);
                            position_contribution = add(position_contribution, scale(gradient, output_adjoint * parameters.masses[other] + density_adjoint[other] * parameters.masses[particle]));
                            mass_contribution += density_adjoint[other] * density_kernel(displacement, support_radius, pbf_kernel);
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, load_boundary_position(boundary, neighbor));
                            const float volume = boundary.volumes[neighbor];
                            const float value = density_kernel(displacement, support_radius, pbf_kernel);
                            position_contribution = add(position_contribution, scale(density_kernel_gradient(displacement, support_radius, pbf_kernel), output_adjoint * parameters.rest_densities[particle] * volume));
                            rest_density_contribution += output_adjoint * volume * value;
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.rest_densities[particle] += rest_density_contribution;
        }

        __global__ void non_pressure_forward_kernel(const std::uint32_t particle_count, const float support_radius, const Float3 gravity, const ConstVector positions, const ConstVector velocities, const ConstVector controls, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const Vector accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 velocity = load(velocities, particle);
            Float3 acceleration = add(gravity, load(controls, particle));
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const float viscosity = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float weight = viscosity * parameters.masses[neighbor] * viscosity_laplacian(displacement, support_radius) / densities[neighbor];
                            acceleration = add(acceleration, scale(subtract(load(velocities, neighbor), velocity), weight));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, load_boundary_position(boundary, neighbor));
                            const float weight = parameters.viscosities[particle] * boundary.volumes[neighbor] * viscosity_laplacian(displacement, support_radius);
                            acceleration = add(acceleration, scale(subtract(load_boundary_velocity(boundary, neighbor), velocity), weight));
                        }
                    }
            store(accelerations, particle, acceleration);
        }

        __global__ void non_pressure_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ConstVector velocities, const ConstVector control_tangent, const ConstVector position_tangent, const ConstVector velocity_tangent, const ParticleParameters parameters, const ParticleParameterTangent parameter_tangent, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const float* density_tangent, const Vector acceleration_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 velocity = load(velocities, particle);
            const Float3 position_dot = load(position_tangent, particle);
            const Float3 velocity_dot = load(velocity_tangent, particle);
            Float3 result = load(control_tangent, particle);
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const float laplacian = viscosity_laplacian(displacement, support_radius);
                            const float laplacian_dot = viscosity_laplacian_tangent(displacement, displacement_dot, support_radius);
                            const float viscosity = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float viscosity_dot = 0.5F * (parameter_tangent.viscosities[particle] + parameter_tangent.viscosities[neighbor]);
                            const float mass = parameters.masses[neighbor];
                            const float density = densities[neighbor];
                            const float weight = viscosity * mass * laplacian / density;
                            const float weight_dot = viscosity_dot * mass * laplacian / density + viscosity * parameter_tangent.masses[neighbor] * laplacian / density + viscosity * mass * laplacian_dot / density - viscosity * mass * laplacian * density_tangent[neighbor] / (density * density);
                            const Float3 difference = subtract(load(velocities, neighbor), velocity);
                            const Float3 difference_dot = subtract(load(velocity_tangent, neighbor), velocity_dot);
                            result = add(result, add(scale(difference, weight_dot), scale(difference_dot, weight)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, load_boundary_position(boundary, neighbor));
                            const float laplacian = viscosity_laplacian(displacement, support_radius);
                            const float weight = parameters.viscosities[particle] * boundary.volumes[neighbor] * laplacian;
                            const float weight_dot = boundary.volumes[neighbor] * (parameter_tangent.viscosities[particle] * laplacian + parameters.viscosities[particle] * viscosity_laplacian_tangent(displacement, position_dot, support_radius));
                            const Float3 difference = subtract(load_boundary_velocity(boundary, neighbor), velocity);
                            result = add(result, add(scale(difference, weight_dot), scale(velocity_dot, -weight)));
                        }
                    }
            store(acceleration_tangent, particle, result);
        }

        __global__ void non_pressure_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ConstVector velocities, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const ConstVectorAdjoint acceleration_adjoint, const VectorAdjoint position_adjoint, const VectorAdjoint velocity_adjoint, const VectorAdjoint control_adjoint, double* density_adjoint, const ParticleParameterAdjoint parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 velocity = load(velocities, particle);
            const Double3 local_adjoint = load(acceleration_adjoint, particle);
            Double3 position_contribution{};
            Double3 velocity_contribution{};
            double density_contribution = 0.0;
            double mass_contribution = 0.0;
            double viscosity_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 difference = subtract(load(velocities, neighbor), velocity);
                            const float laplacian = viscosity_laplacian(displacement, support_radius);
                            const float viscosity = 0.5F * (parameters.viscosities[particle] + parameters.viscosities[neighbor]);
                            const float mass = parameters.masses[neighbor];
                            const float density = densities[neighbor];
                            const float weight = viscosity * mass * laplacian / density;
                            const double weight_adjoint = dot(local_adjoint, difference);
                            velocity_contribution = add(velocity_contribution, scale(local_adjoint, -weight));
                            const double viscosity_adjoint = weight_adjoint * mass * laplacian / density;
                            viscosity_contribution += 0.5 * viscosity_adjoint;
                            position_contribution = add(position_contribution, viscosity_laplacian_gradient(displacement, support_radius, weight_adjoint * viscosity * mass / density));

                            const Double3 cross_adjoint = load(acceleration_adjoint, neighbor);
                            const float cross_weight = viscosity * parameters.masses[particle] * laplacian / densities[particle];
                            const double cross_weight_adjoint = dot(cross_adjoint, scale(difference, -1.0F));
                            velocity_contribution = add(velocity_contribution, scale(cross_adjoint, cross_weight));
                            mass_contribution += cross_weight_adjoint * viscosity * laplacian / densities[particle];
                            density_contribution -= cross_weight_adjoint * viscosity * parameters.masses[particle] * laplacian / (densities[particle] * densities[particle]);
                            viscosity_contribution += 0.5 * cross_weight_adjoint * parameters.masses[particle] * laplacian / densities[particle];
                            position_contribution = add(position_contribution, viscosity_laplacian_gradient(displacement, support_radius, cross_weight_adjoint * viscosity * parameters.masses[particle] / densities[particle]));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, load_boundary_position(boundary, neighbor));
                            const Float3 difference = subtract(load_boundary_velocity(boundary, neighbor), velocity);
                            const float laplacian = viscosity_laplacian(displacement, support_radius);
                            const float weight = parameters.viscosities[particle] * boundary.volumes[neighbor] * laplacian;
                            const double weight_adjoint = dot(local_adjoint, difference);
                            velocity_contribution = add(velocity_contribution, scale(local_adjoint, -weight));
                            viscosity_contribution += weight_adjoint * boundary.volumes[neighbor] * laplacian;
                            position_contribution = add(position_contribution, viscosity_laplacian_gradient(displacement, support_radius, weight_adjoint * parameters.viscosities[particle] * boundary.volumes[neighbor]));
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            accumulate(velocity_adjoint, particle, velocity_contribution);
            accumulate(control_adjoint, particle, local_adjoint);
            density_adjoint[particle] += density_contribution;
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.viscosities[particle] += viscosity_contribution;
        }

        __global__ void pressure_forward_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const float* pressures, const Vector accelerations) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            Float3 acceleration{};
            const float first_term = pressures[particle] / (densities[particle] * densities[particle]);
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const float second_term = pressures[neighbor] / (densities[neighbor] * densities[neighbor]);
                            acceleration = add(acceleration, scale(cubic_gradient(subtract(position, load(positions, neighbor)), support_radius), -parameters.masses[neighbor] * (first_term + second_term)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            acceleration = add(acceleration, scale(cubic_gradient(subtract(position, load_boundary_position(boundary, neighbor)), support_radius), -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term));
                        }
                    }
            store(accelerations, particle, acceleration);
        }

        __global__ void pressure_jvp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ConstVector position_tangent, const ParticleParameters parameters, const ParticleParameterTangent parameter_tangent, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const float* density_tangent, const float* pressures, const float* pressure_tangent, const Vector acceleration_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Float3 position_dot = load(position_tangent, particle);
            const float density = densities[particle];
            const float first_term = pressures[particle] / (density * density);
            const float first_term_dot = pressure_tangent[particle] / (density * density) - 2.0F * pressures[particle] * density_tangent[particle] / (density * density * density);
            Float3 result{};
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const float neighbor_density = densities[neighbor];
                            const float second_term = pressures[neighbor] / (neighbor_density * neighbor_density);
                            const float second_term_dot = pressure_tangent[neighbor] / (neighbor_density * neighbor_density) - 2.0F * pressures[neighbor] * density_tangent[neighbor] / (neighbor_density * neighbor_density * neighbor_density);
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 displacement_dot = subtract(position_dot, load(position_tangent, neighbor));
                            const float factor = -parameters.masses[neighbor] * (first_term + second_term);
                            const float factor_dot = -parameter_tangent.masses[neighbor] * (first_term + second_term) - parameters.masses[neighbor] * (first_term_dot + second_term_dot);
                            result = add(result, add(scale(cubic_gradient(displacement, support_radius), factor_dot), scale(cubic_gradient_tangent(displacement, displacement_dot, support_radius), factor)));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, load_boundary_position(boundary, neighbor));
                            const float factor = -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term;
                            const float factor_dot = -boundary.volumes[neighbor] * (parameter_tangent.rest_densities[particle] * first_term + parameters.rest_densities[particle] * first_term_dot);
                            result = add(result, add(scale(cubic_gradient(displacement, support_radius), factor_dot), scale(cubic_gradient_tangent(displacement, position_dot, support_radius), factor)));
                        }
                    }
            store(acceleration_tangent, particle, result);
        }

        __global__ void pressure_vjp_kernel(const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const float* pressures, const ConstVectorAdjoint acceleration_adjoint, const VectorAdjoint position_adjoint, double* density_adjoint, double* pressure_adjoint, const ParticleParameterAdjoint parameter_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 position = load(positions, particle);
            const Double3 local_adjoint = load(acceleration_adjoint, particle);
            const double first_term = static_cast<double>(pressures[particle]) / (densities[particle] * densities[particle]);
            Double3 position_contribution{};
            double first_term_adjoint = 0.0;
            double mass_contribution = 0.0;
            double rest_density_contribution = 0.0;
            int cell_x, cell_y, cell_z;
            particle_cell(neighborhood, position, cell_x, cell_y, cell_z);
            for (int offset_x = -1; offset_x <= 1; ++offset_x)
                for (int offset_y = -1; offset_y <= 1; ++offset_y)
                    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
                        const CellRange range = cell_range(neighborhood, cell_x + offset_x, cell_y + offset_y, cell_z + offset_z);
                        if (!range.valid) continue;
                        const std::uint32_t first = range.first;
                        const std::uint32_t last = range.last;
                        for (std::uint32_t sorted = first; sorted < last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_particle_indices[sorted];
                            if (neighbor == particle) continue;
                            const Float3 displacement = subtract(position, load(positions, neighbor));
                            const Float3 gradient = cubic_gradient(displacement, support_radius);
                            const double second_term = static_cast<double>(pressures[neighbor]) / (densities[neighbor] * densities[neighbor]);
                            const double sum = first_term + second_term;
                            const double local_factor = -parameters.masses[neighbor] * sum;
                            const double local_factor_adjoint = dot(local_adjoint, gradient);
                            first_term_adjoint -= parameters.masses[neighbor] * local_factor_adjoint;
                            position_contribution = add(position_contribution, cubic_hessian_product(displacement, scale(local_adjoint, local_factor), support_radius));

                            const Double3 cross_adjoint = load(acceleration_adjoint, neighbor);
                            const double cross_projection = dot(cross_adjoint, gradient);
                            mass_contribution += sum * cross_projection;
                            first_term_adjoint += parameters.masses[particle] * cross_projection;
                            position_contribution = add(position_contribution, cubic_hessian_product(displacement, scale(cross_adjoint, parameters.masses[particle] * sum), support_radius));
                        }
                        const std::uint32_t boundary_first = range.boundary_first;
                        const std::uint32_t boundary_last = range.boundary_last;
                        for (std::uint32_t sorted = boundary_first; sorted < boundary_last; ++sorted) {
                            const std::uint32_t neighbor = neighborhood.sorted_boundary_indices[sorted];
                            const Float3 displacement = subtract(position, load_boundary_position(boundary, neighbor));
                            const Float3 gradient = cubic_gradient(displacement, support_radius);
                            const double factor = -parameters.rest_densities[particle] * boundary.volumes[neighbor] * first_term;
                            const double factor_adjoint = dot(local_adjoint, gradient);
                            rest_density_contribution -= factor_adjoint * boundary.volumes[neighbor] * first_term;
                            first_term_adjoint -= factor_adjoint * parameters.rest_densities[particle] * boundary.volumes[neighbor];
                            position_contribution = add(position_contribution, cubic_hessian_product(displacement, scale(local_adjoint, factor), support_radius));
                        }
                    }
            accumulate(position_adjoint, particle, position_contribution);
            parameter_adjoint.masses[particle] += mass_contribution;
            parameter_adjoint.rest_densities[particle] += rest_density_contribution;
            pressure_adjoint[particle] += first_term_adjoint / (densities[particle] * densities[particle]);
            density_adjoint[particle] -= 2.0 * first_term_adjoint * pressures[particle] / (densities[particle] * densities[particle] * densities[particle]);
        }

        __device__ void collision_mask(const Domain domain, const Float3 predicted_position, bool& collision_x, bool& collision_y, bool& collision_z) {
            collision_x = predicted_position.x < domain.minimum_x || predicted_position.x > domain.maximum_x;
            collision_y = predicted_position.y < domain.minimum_y || predicted_position.y > domain.maximum_y;
            collision_z = predicted_position.z < domain.minimum_z || predicted_position.z > domain.maximum_z;
        }

        __global__ void integrate_forward_kernel(const std::uint32_t particle_count, const float time_step, const Domain domain, const ConstVector positions, const ConstVector velocities, const ConstVector accelerations, const Vector next_positions, const Vector next_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            Float3 velocity = add(load(velocities, particle), scale(load(accelerations, particle), time_step));
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
            store(next_positions, particle, position);
            store(next_velocities, particle, velocity);
        }

        __global__ void integrate_jvp_kernel(const std::uint32_t particle_count, const float time_step, const Domain domain, const ConstVector positions, const ConstVector velocities, const ConstVector accelerations, const ConstVector position_tangent, const ConstVector velocity_tangent, const ConstVector acceleration_tangent, const Vector next_position_tangent, const Vector next_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(load(accelerations, particle), time_step));
            const Float3 predicted_position = add(load(positions, particle), scale(predicted_velocity, time_step));
            Float3 velocity_dot = add(load(velocity_tangent, particle), scale(load(acceleration_tangent, particle), time_step));
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
            store(next_position_tangent, particle, position_dot);
            store(next_velocity_tangent, particle, velocity_dot);
        }

        __global__ void integrate_vjp_kernel(const std::uint32_t particle_count, const float time_step, const Domain domain, const ConstVector positions, const ConstVector velocities, const ConstVector accelerations, const ConstVectorAdjoint next_position_adjoint, const ConstVectorAdjoint next_velocity_adjoint, const VectorAdjoint position_adjoint, const VectorAdjoint velocity_adjoint, const VectorAdjoint acceleration_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Float3 predicted_velocity = add(load(velocities, particle), scale(load(accelerations, particle), time_step));
            const Float3 predicted_position = add(load(positions, particle), scale(predicted_velocity, time_step));
            bool collision_x, collision_y, collision_z;
            collision_mask(domain, predicted_position, collision_x, collision_y, collision_z);
            double position_bar_x = collision_x ? 0.0 : next_position_adjoint.x[particle];
            double position_bar_y = collision_y ? 0.0 : next_position_adjoint.y[particle];
            double position_bar_z = collision_z ? 0.0 : next_position_adjoint.z[particle];
            double velocity_bar_x = next_velocity_adjoint.x[particle] + time_step * position_bar_x;
            double velocity_bar_y = next_velocity_adjoint.y[particle] + time_step * position_bar_y;
            double velocity_bar_z = next_velocity_adjoint.z[particle] + time_step * position_bar_z;
            if (domain.no_slip != 0u && (collision_x || collision_y || collision_z)) velocity_bar_x = velocity_bar_y = velocity_bar_z = 0.0;
            else {
                if (collision_x) velocity_bar_x = 0.0;
                if (collision_y) velocity_bar_y = 0.0;
                if (collision_z) velocity_bar_z = 0.0;
            }
            accumulate(position_adjoint, particle, {position_bar_x, position_bar_y, position_bar_z});
            accumulate(velocity_adjoint, particle, {velocity_bar_x, velocity_bar_y, velocity_bar_z});
            accumulate(acceleration_adjoint, particle, {time_step * velocity_bar_x, time_step * velocity_bar_y, time_step * velocity_bar_z});
        }

        __global__ void add_kernel(const std::uint32_t particle_count, const ConstVector first, const ConstVector second, const Vector output) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            output.x[particle] = first.x[particle] + second.x[particle];
            output.y[particle] = first.y[particle] + second.y[particle];
            output.z[particle] = first.z[particle] + second.z[particle];
        }

        __global__ void add_adjoint_kernel(const std::uint32_t particle_count, const ConstVectorAdjoint output_adjoint, const VectorAdjoint first_adjoint, const VectorAdjoint second_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const double x = output_adjoint.x[particle];
            const double y = output_adjoint.y[particle];
            const double z = output_adjoint.z[particle];
            first_adjoint.x[particle] += x;
            first_adjoint.y[particle] += y;
            first_adjoint.z[particle] += z;
            second_adjoint.x[particle] += x;
            second_adjoint.y[particle] += y;
            second_adjoint.z[particle] += z;
        }

    } // namespace

    void launch_copy_vector(void* stream, const std::uint32_t count, const ConstVector source, const Vector destination) {
        copy_vector_kernel<<<blocks(count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(count, source, destination);
        check_launch("launch_copy_vector");
    }

    void launch_copy_vector_adjoint(void* stream, const std::uint32_t count, const ConstVectorAdjoint source, const VectorAdjoint destination) {
        copy_vector_adjoint_kernel<<<blocks(count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(count, source, destination);
        check_launch("launch_copy_vector_adjoint");
    }

    void launch_accumulate_vector_adjoint(void* stream, const std::uint32_t count, const ConstVectorAdjoint source, const VectorAdjoint destination) {
        accumulate_vector_adjoint_kernel<<<blocks(count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(count, source, destination);
        check_launch("launch_accumulate_vector_adjoint");
    }

    void launch_copy_scalar(void* stream, const std::uint32_t count, const float* source, float* destination) {
        copy_scalar_kernel<<<blocks(count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(count, source, destination);
        check_launch("launch_copy_scalar");
    }

    void launch_copy_scalar_adjoint(void* stream, const std::uint32_t count, const double* source, double* destination) {
        copy_scalar_kernel<<<blocks(count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(count, source, destination);
        check_launch("launch_copy_scalar_adjoint");
    }

    void launch_accumulate_scalar_adjoint(void* stream, const std::uint32_t count, const double* source, double* destination) {
        accumulate_scalar_adjoint_kernel<<<blocks(count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(count, source, destination);
        check_launch("launch_accumulate_scalar_adjoint");
    }

    void query_neighbor_storage(const std::uint32_t particle_count, std::size_t& sort_bytes) {
        cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, static_cast<std::uint64_t*>(nullptr), static_cast<std::uint64_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), particle_count);
    }

    void launch_build_neighborhood(void* stream, const std::uint32_t particle_count, const std::uint32_t boundary_count, const float support_radius, const float time_step, const std::uint64_t step_index, const Domain domain, const ConstVector positions, const ConstVector boundary_positions, const ConstVector boundary_velocities, std::uint64_t* unsorted_keys, std::uint32_t* unsorted_particle_indices, std::uint64_t* unsorted_boundary_keys, std::uint32_t* unsorted_boundary_indices, void* sort_scratch, const std::size_t sort_scratch_bytes, void* boundary_sort_scratch, const std::size_t boundary_sort_scratch_bytes, std::uint64_t* sorted_keys, std::uint32_t* sorted_particle_indices, std::uint32_t* cell_offsets, std::uint64_t* sorted_boundary_keys, std::uint32_t* sorted_boundary_indices, std::uint32_t* boundary_cell_offsets) {
        const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const float time = static_cast<float>(step_index) * time_step;
        key_kernel<<<blocks(particle_count), block_size, 0, cuda_stream>>>(particle_count, support_radius, domain, time, 0u, positions, {}, unsorted_keys, unsorted_particle_indices);
        std::size_t particle_sort_bytes = sort_scratch_bytes;
        cub::DeviceRadixSort::SortPairs(sort_scratch, particle_sort_bytes, unsorted_keys, sorted_keys, unsorted_particle_indices, sorted_particle_indices, particle_count, 0, 64, cuda_stream);
        const std::uint32_t cells_x = static_cast<std::uint32_t>(ceilf((domain.maximum_x - domain.minimum_x) / support_radius));
        const std::uint32_t cells_y = static_cast<std::uint32_t>(ceilf((domain.maximum_y - domain.minimum_y) / support_radius));
        const std::uint32_t cells_z = static_cast<std::uint32_t>(ceilf((domain.maximum_z - domain.minimum_z) / support_radius));
        const std::uint32_t cell_count = cells_x * cells_y * cells_z;
        cell_offsets_kernel<<<blocks(cell_count + 1u), block_size, 0, cuda_stream>>>(particle_count, sorted_keys, cell_count, cell_offsets);
        if (boundary_count != 0u) {
            key_kernel<<<blocks(boundary_count), block_size, 0, cuda_stream>>>(boundary_count, support_radius, domain, time, 1u, boundary_positions, boundary_velocities, unsorted_boundary_keys, unsorted_boundary_indices);
            std::size_t boundary_bytes = boundary_sort_scratch_bytes;
            cub::DeviceRadixSort::SortPairs(boundary_sort_scratch, boundary_bytes, unsorted_boundary_keys, sorted_boundary_keys, unsorted_boundary_indices, sorted_boundary_indices, boundary_count, 0, 64, cuda_stream);
            cell_offsets_kernel<<<blocks(cell_count + 1u), block_size, 0, cuda_stream>>>(boundary_count, sorted_boundary_keys, cell_count, boundary_cell_offsets);
        } else cudaMemsetAsync(boundary_cell_offsets, 0, (static_cast<std::size_t>(cell_count) + 1u) * sizeof(std::uint32_t), cuda_stream);
        check_launch("launch_build_neighborhood");
    }

    void launch_density_forward(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector topology_positions, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, float* densities) {
        density_forward_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, 0u, topology_positions, positions, parameters, neighborhood, boundary, densities);
        check_launch("launch_density_forward");
    }

    void launch_density_jvp(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector topology_positions, const ConstVector positions, const ConstVector position_tangent, const ParticleParameters parameters, const ParticleParameterTangent parameter_tangent, const Neighborhood neighborhood, const Boundary boundary, float* density_tangent) {
        density_jvp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, 0u, topology_positions, positions, position_tangent, parameters, parameter_tangent, neighborhood, boundary, density_tangent);
        check_launch("launch_density_jvp");
    }

    void launch_density_vjp(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector topology_positions, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const double* density_adjoint, const VectorAdjoint position_adjoint, const ParticleParameterAdjoint parameter_adjoint) {
        density_vjp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, 0u, topology_positions, positions, parameters, neighborhood, boundary, density_adjoint, position_adjoint, parameter_adjoint);
        check_launch("launch_density_vjp");
    }

    void launch_pbf_density_forward(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector topology_positions, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, float* densities) {
        density_forward_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, 1u, topology_positions, positions, parameters, neighborhood, boundary, densities);
        check_launch("launch_pbf_density_forward");
    }

    void launch_pbf_density_jvp(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector topology_positions, const ConstVector positions, const ConstVector position_tangent, const ParticleParameters parameters, const ParticleParameterTangent parameter_tangent, const Neighborhood neighborhood, const Boundary boundary, float* density_tangent) {
        density_jvp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, 1u, topology_positions, positions, position_tangent, parameters, parameter_tangent, neighborhood, boundary, density_tangent);
        check_launch("launch_pbf_density_jvp");
    }

    void launch_pbf_density_vjp(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector topology_positions, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const double* density_adjoint, const VectorAdjoint position_adjoint, const ParticleParameterAdjoint parameter_adjoint) {
        density_vjp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, 1u, topology_positions, positions, parameters, neighborhood, boundary, density_adjoint, position_adjoint, parameter_adjoint);
        check_launch("launch_pbf_density_vjp");
    }

    void launch_non_pressure_forward(void* stream, const std::uint32_t particle_count, const float support_radius, const float gravity_x, const float gravity_y, const float gravity_z, const ConstVector positions, const ConstVector velocities, const ConstVector external_accelerations, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const Vector accelerations) {
        non_pressure_forward_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, {gravity_x, gravity_y, gravity_z}, positions, velocities, external_accelerations, parameters, neighborhood, boundary, densities, accelerations);
        check_launch("launch_non_pressure_forward");
    }

    void launch_non_pressure_jvp(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ConstVector velocities, const ConstVector external_acceleration_tangent, const ConstVector position_tangent, const ConstVector velocity_tangent, const ParticleParameters parameters, const ParticleParameterTangent parameter_tangent, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const float* density_tangent, const Vector acceleration_tangent) {
        non_pressure_jvp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, positions, velocities, external_acceleration_tangent, position_tangent, velocity_tangent, parameters, parameter_tangent, neighborhood, boundary, densities, density_tangent, acceleration_tangent);
        check_launch("launch_non_pressure_jvp");
    }

    void launch_non_pressure_vjp(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ConstVector velocities, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const ConstVectorAdjoint acceleration_adjoint, const VectorAdjoint position_adjoint, const VectorAdjoint velocity_adjoint, const VectorAdjoint control_adjoint, double* density_adjoint, const ParticleParameterAdjoint parameter_adjoint) {
        non_pressure_vjp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, positions, velocities, parameters, neighborhood, boundary, densities, acceleration_adjoint, position_adjoint, velocity_adjoint, control_adjoint, density_adjoint, parameter_adjoint);
        check_launch("launch_non_pressure_vjp");
    }

    void launch_integrate_forward(void* stream, const std::uint32_t particle_count, const float time_step, const Domain domain, const ConstVector positions, const ConstVector velocities, const ConstVector accelerations, const Vector next_positions, const Vector next_velocities) {
        integrate_forward_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, domain, positions, velocities, accelerations, next_positions, next_velocities);
        check_launch("launch_integrate_forward");
    }

    void launch_integrate_jvp(void* stream, const std::uint32_t particle_count, const float time_step, const Domain domain, const ConstVector positions, const ConstVector velocities, const ConstVector accelerations, const ConstVector position_tangent, const ConstVector velocity_tangent, const ConstVector acceleration_tangent, const Vector next_position_tangent, const Vector next_velocity_tangent) {
        integrate_jvp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, domain, positions, velocities, accelerations, position_tangent, velocity_tangent, acceleration_tangent, next_position_tangent, next_velocity_tangent);
        check_launch("launch_integrate_jvp");
    }

    void launch_integrate_vjp(void* stream, const std::uint32_t particle_count, const float time_step, const Domain domain, const ConstVector positions, const ConstVector velocities, const ConstVector accelerations, const ConstVectorAdjoint next_position_adjoint, const ConstVectorAdjoint next_velocity_adjoint, const VectorAdjoint position_adjoint, const VectorAdjoint velocity_adjoint, const VectorAdjoint acceleration_adjoint) {
        integrate_vjp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, time_step, domain, positions, velocities, accelerations, next_position_adjoint, next_velocity_adjoint, position_adjoint, velocity_adjoint, acceleration_adjoint);
        check_launch("launch_integrate_vjp");
    }

    void launch_pressure_acceleration_forward(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const float* pressures, const Vector accelerations) {
        pressure_forward_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, positions, parameters, neighborhood, boundary, densities, pressures, accelerations);
        check_launch("launch_pressure_acceleration_forward");
    }

    void launch_pressure_acceleration_jvp(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ConstVector position_tangent, const ParticleParameters parameters, const ParticleParameterTangent parameter_tangent, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const float* density_tangent, const float* pressures, const float* pressure_tangent, const Vector acceleration_tangent) {
        pressure_jvp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, positions, position_tangent, parameters, parameter_tangent, neighborhood, boundary, densities, density_tangent, pressures, pressure_tangent, acceleration_tangent);
        check_launch("launch_pressure_acceleration_jvp");
    }

    void launch_pressure_acceleration_vjp(void* stream, const std::uint32_t particle_count, const float support_radius, const ConstVector positions, const ParticleParameters parameters, const Neighborhood neighborhood, const Boundary boundary, const float* densities, const float* pressures, const ConstVectorAdjoint acceleration_adjoint, const VectorAdjoint position_adjoint, double* density_adjoint, double* pressure_adjoint, const ParticleParameterAdjoint parameter_adjoint) {
        pressure_vjp_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, support_radius, positions, parameters, neighborhood, boundary, densities, pressures, acceleration_adjoint, position_adjoint, density_adjoint, pressure_adjoint, parameter_adjoint);
        check_launch("launch_pressure_acceleration_vjp");
    }

    void launch_add_acceleration(void* stream, const std::uint32_t particle_count, const ConstVector first, const ConstVector second, const Vector output) {
        add_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, first, second, output);
        check_launch("launch_add_acceleration");
    }

    void launch_add_acceleration_adjoint(void* stream, const std::uint32_t particle_count, const ConstVectorAdjoint output_adjoint, const VectorAdjoint first_adjoint, const VectorAdjoint second_adjoint) {
        add_adjoint_kernel<<<blocks(particle_count), block_size, 0, static_cast<cudaStream_t>(stream)>>>(particle_count, output_adjoint, first_adjoint, second_adjoint);
        check_launch("launch_add_acceleration_adjoint");
    }

} // namespace xayah::fluid::sph::cuda_kernel

