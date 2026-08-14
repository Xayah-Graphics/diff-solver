#include "grid.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace xayah::fluid::grid::cuda_kernels {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        constexpr std::uint64_t invalid_key = ~std::uint64_t{};

        struct Vector {
            float x;
            float y;
            float z;
        };

        struct Weight {
            float value;
            Vector gradient;
            Vector offset;
            std::uint64_t face;
            bool valid;
        };

        __host__ std::uint32_t blocks(const std::size_t count) {
            return static_cast<std::uint32_t>((count + block_size - 1u) / block_size);
        }

        __host__ void check(const cudaError_t status, const char* operation) {
            if (status != cudaSuccess) throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorString(status));
        }

        __host__ void check_launch(const char* operation) {
            check(cudaGetLastError(), operation);
        }

        __host__ __device__ std::uint64_t cells(const Configuration configuration) {
            return static_cast<std::uint64_t>(configuration.nx) * configuration.ny * configuration.nz;
        }

        __host__ __device__ int extent(const Configuration configuration, const int axis, const int dimension) {
            const int base = dimension == 0 ? configuration.nx : dimension == 1 ? configuration.ny : configuration.nz;
            return base + (axis == dimension ? 1 : 0);
        }

        __host__ __device__ std::uint64_t face_count(const Configuration configuration, const int axis) {
            return static_cast<std::uint64_t>(extent(configuration, axis, 0)) * extent(configuration, axis, 1) * extent(configuration, axis, 2);
        }

        __host__ __device__ std::uint64_t face_offset(const Configuration configuration, const int axis) {
            if (axis == 0) return 0u;
            if (axis == 1) return face_count(configuration, 0);
            return face_count(configuration, 0) + face_count(configuration, 1);
        }

        __host__ __device__ std::uint64_t total_faces(const Configuration configuration) {
            return face_count(configuration, 0) + face_count(configuration, 1) + face_count(configuration, 2);
        }

        __host__ __device__ std::uint64_t index3(const int x, const int y, const int z, const int nx, const int ny) {
            return static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(nx) * (static_cast<std::uint64_t>(y) + static_cast<std::uint64_t>(ny) * z);
        }

        __host__ __device__ void decode(const std::uint64_t index, const int nx, const int ny, int& x, int& y, int& z) {
            x = static_cast<int>(index % nx);
            const std::uint64_t yz = index / nx;
            y = static_cast<int>(yz % ny);
            z = static_cast<int>(yz / ny);
        }

        __host__ __device__ float* component(const VectorField field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const float* component(const ConstVectorField field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ double* component(const VectorAdjointField field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const double* component(const ConstVectorAdjointField field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ float* component(const StaggeredField field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const float* component(const ConstStaggeredField field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ double* component(const StaggeredAdjointField field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const double* component(const ConstStaggeredAdjointField field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __device__ Vector load(const ConstVectorField field, const std::uint32_t index) {
            return {.x = field.x[index], .y = field.y[index], .z = field.z[index]};
        }

        __device__ void store(const VectorField field, const std::uint32_t index, const Vector value) {
            field.x[index] = value.x;
            field.y[index] = value.y;
            field.z[index] = value.z;
        }

        __device__ float& component(Vector& vector, const int axis) {
            return axis == 0 ? vector.x : axis == 1 ? vector.y : vector.z;
        }

        __device__ Vector operator+(const Vector left, const Vector right) {
            return {.x = left.x + right.x, .y = left.y + right.y, .z = left.z + right.z};
        }

        __device__ Vector operator-(const Vector left, const Vector right) {
            return {.x = left.x - right.x, .y = left.y - right.y, .z = left.z - right.z};
        }

        __device__ Vector operator*(const float scale, const Vector vector) {
            return {.x = scale * vector.x, .y = scale * vector.y, .z = scale * vector.z};
        }

        __device__ float dot(const Vector left, const Vector right) {
            return left.x * right.x + left.y * right.y + left.z * right.z;
        }

        __device__ std::uint32_t cell_type(const std::uint32_t* types, const Configuration configuration, const int x, const int y, const int z) {
            if (x < 0 || y < 0 || z < 0 || x >= static_cast<int>(configuration.nx) || y >= static_cast<int>(configuration.ny) || z >= static_cast<int>(configuration.nz)) return 2u;
            return types[index3(x, y, z, configuration.nx, configuration.ny)];
        }

        __device__ Weight weight(const Configuration configuration, const Vector position, const int axis, const int slot) {
            const float inverse_h = 1.0F / configuration.cell_size;
            const Vector grid_offset{
                .x = axis == 0 ? 0.0F : 0.5F,
                .y = axis == 1 ? 0.0F : 0.5F,
                .z = axis == 2 ? 0.0F : 0.5F,
            };
            const Vector q{
                .x = (position.x - configuration.origin_x) * inverse_h - grid_offset.x,
                .y = (position.y - configuration.origin_y) * inverse_h - grid_offset.y,
                .z = (position.z - configuration.origin_z) * inverse_h - grid_offset.z,
            };
            const int base_x = static_cast<int>(floorf(q.x - 0.5F));
            const int base_y = static_cast<int>(floorf(q.y - 0.5F));
            const int base_z = static_cast<int>(floorf(q.z - 0.5F));
            const int local_x = slot % 3;
            const int local_y = slot / 3 % 3;
            const int local_z = slot / 9;
            const float fx = q.x - base_x;
            const float fy = q.y - base_y;
            const float fz = q.z - base_z;
            const float wx[3]{0.5F * (1.5F - fx) * (1.5F - fx), 0.75F - (fx - 1.0F) * (fx - 1.0F), 0.5F * (fx - 0.5F) * (fx - 0.5F)};
            const float wy[3]{0.5F * (1.5F - fy) * (1.5F - fy), 0.75F - (fy - 1.0F) * (fy - 1.0F), 0.5F * (fy - 0.5F) * (fy - 0.5F)};
            const float wz[3]{0.5F * (1.5F - fz) * (1.5F - fz), 0.75F - (fz - 1.0F) * (fz - 1.0F), 0.5F * (fz - 0.5F) * (fz - 0.5F)};
            const float dx[3]{fx - 1.5F, -2.0F * (fx - 1.0F), fx - 0.5F};
            const float dy[3]{fy - 1.5F, -2.0F * (fy - 1.0F), fy - 0.5F};
            const float dz[3]{fz - 1.5F, -2.0F * (fz - 1.0F), fz - 0.5F};
            const int x = base_x + local_x;
            const int y = base_y + local_y;
            const int z = base_z + local_z;
            const int nx = extent(configuration, axis, 0);
            const int ny = extent(configuration, axis, 1);
            const int nz = extent(configuration, axis, 2);
            const bool valid = x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
            const float value = wx[local_x] * wy[local_y] * wz[local_z];
            const Vector gradient{
                .x = inverse_h * dx[local_x] * wy[local_y] * wz[local_z],
                .y = inverse_h * wx[local_x] * dy[local_y] * wz[local_z],
                .z = inverse_h * wx[local_x] * wy[local_y] * dz[local_z],
            };
            const Vector face_position{
                .x = configuration.origin_x + (x + grid_offset.x) * configuration.cell_size,
                .y = configuration.origin_y + (y + grid_offset.y) * configuration.cell_size,
                .z = configuration.origin_z + (z + grid_offset.z) * configuration.cell_size,
            };
            return {.value = value, .gradient = gradient, .offset = face_position - position, .face = valid ? face_offset(configuration, axis) + index3(x, y, z, nx, ny) : invalid_key, .valid = valid};
        }

        __device__ float affine_velocity(const ConstMatrixField affine, const std::uint32_t particle, const int axis, const Vector offset) {
            return affine.values[axis * 3][particle] * offset.x + affine.values[axis * 3 + 1][particle] * offset.y + affine.values[axis * 3 + 2][particle] * offset.z;
        }

        __device__ std::size_t lower_bound(const std::uint64_t* keys, const std::size_t count, const std::uint64_t target) {
            std::size_t first = 0u;
            std::size_t length = count;
            while (length != 0u) {
                const std::size_t half = length / 2u;
                const std::size_t middle = first + half;
                if (keys[middle] < target) {
                    first = middle + 1u;
                    length -= half + 1u;
                } else {
                    length = half;
                }
            }
            return first;
        }

        __global__ void marker_kernel(const Configuration configuration, const std::uint32_t particle_count, const std::uint64_t* sorted_cell_keys, const std::uint32_t* domain_types, std::uint32_t* types) {
            const std::uint64_t cell = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cell >= cells(configuration)) return;
            if (domain_types[cell] == 2u) {
                types[cell] = 2u;
                return;
            }
            const std::size_t entry = lower_bound(sorted_cell_keys, particle_count, cell);
            types[cell] = entry < particle_count && sorted_cell_keys[entry] == cell ? 1u : 0u;
        }

        __global__ void cell_topology_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, std::uint64_t* keys, std::uint32_t* ids) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            const int x = static_cast<int>(floorf((position.x - configuration.origin_x) / configuration.cell_size));
            const int y = static_cast<int>(floorf((position.y - configuration.origin_y) / configuration.cell_size));
            const int z = static_cast<int>(floorf((position.z - configuration.origin_z) / configuration.cell_size));
            keys[particle] = x >= 0 && y >= 0 && z >= 0 && x < static_cast<int>(configuration.nx) && y < static_cast<int>(configuration.ny) && z < static_cast<int>(configuration.nz) ? index3(x, y, z, configuration.nx, configuration.ny) : invalid_key;
            ids[particle] = particle;
        }

        __global__ void p2g_topology_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, std::uint64_t* keys, std::uint32_t* ids) {
            const std::uint64_t contribution = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (contribution >= static_cast<std::uint64_t>(particle_count) * 81u) return;
            const std::uint32_t particle = static_cast<std::uint32_t>(contribution / 81u);
            const int local = static_cast<int>(contribution % 81u);
            const int axis = local / 27;
            const int slot = local % 27;
            keys[contribution] = weight(configuration, load(positions, particle), axis, slot).face;
            ids[contribution] = static_cast<std::uint32_t>(contribution);
        }

        __global__ void p2g_forward_kernel(const Configuration configuration, const std::size_t contribution_count, const bool apic, const ConstVectorField positions, const ConstVectorField velocities, const ConstMatrixField affine, const ConstVectorField accelerations, const float* masses, const float gravity_x, const float gravity_y, const float gravity_z, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, float* face_mass, const StaggeredField old_velocity, const StaggeredField forced_velocity) {
            const std::uint64_t face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (face >= total_faces(configuration)) return;
            const std::size_t begin = lower_bound(sorted_keys, contribution_count, face);
            float mass_sum = 0.0F;
            float momentum_sum = 0.0F;
            float acceleration_sum = 0.0F;
            for (std::size_t entry = begin; entry < contribution_count && sorted_keys[entry] == face; ++entry) {
                const std::uint32_t id = sorted_ids[entry];
                const std::uint32_t particle = id / 81u;
                const int local = id % 81u;
                const int axis = local / 27;
                const Weight sample = weight(configuration, load(positions, particle), axis, local % 27);
                const float mass_weight = masses[particle] * sample.value;
                const float velocity = component(velocities, axis)[particle] + (apic ? affine_velocity(affine, particle, axis, sample.offset) : 0.0F);
                const float gravity = axis == 0 ? gravity_x : axis == 1 ? gravity_y : gravity_z;
                const float acceleration = component(accelerations, axis)[particle] + gravity;
                mass_sum += mass_weight;
                momentum_sum += mass_weight * velocity;
                acceleration_sum += mass_weight * acceleration;
            }
            face_mass[face] = static_cast<float>(mass_sum);
            int axis = face < face_offset(configuration, 1) ? 0 : face < face_offset(configuration, 2) ? 1 : 2;
            const std::uint64_t local_face = face - face_offset(configuration, axis);
            if (mass_sum == 0.0F) {
                component(old_velocity, axis)[local_face] = 0.0F;
                component(forced_velocity, axis)[local_face] = 0.0F;
                return;
            }
            component(old_velocity, axis)[local_face] = momentum_sum / mass_sum;
            component(forced_velocity, axis)[local_face] = (momentum_sum + configuration.time_step * acceleration_sum) / mass_sum;
        }

        __global__ void p2g_jvp_kernel(const Configuration configuration, const std::size_t contribution_count, const bool apic, const ConstVectorField positions, const ConstVectorField velocities, const ConstMatrixField affine, const ConstVectorField accelerations, const float* masses, const float gravity_x, const float gravity_y, const float gravity_z, const ConstVectorField position_tangent, const ConstVectorField velocity_tangent, const ConstMatrixField affine_tangent, const ConstVectorField acceleration_tangent, const float* mass_tangent, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const float* face_mass, const ConstStaggeredField old_velocity, const ConstStaggeredField forced_velocity, const StaggeredField old_velocity_tangent, const StaggeredField forced_velocity_tangent) {
            const std::uint64_t face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (face >= total_faces(configuration)) return;
            const std::size_t begin = lower_bound(sorted_keys, contribution_count, face);
            float mass_sum_tangent = 0.0F;
            float momentum_sum_tangent = 0.0F;
            float acceleration_sum_tangent = 0.0F;
            for (std::size_t entry = begin; entry < contribution_count && sorted_keys[entry] == face; ++entry) {
                const std::uint32_t id = sorted_ids[entry];
                const std::uint32_t particle = id / 81u;
                const int local = id % 81u;
                const int axis = local / 27;
                const Vector position_delta = load(position_tangent, particle);
                const Weight sample = weight(configuration, load(positions, particle), axis, local % 27);
                const float weight_tangent = dot(sample.gradient, position_delta);
                const float mass_weight = masses[particle] * sample.value;
                const float mass_weight_tangent = mass_tangent[particle] * sample.value + masses[particle] * weight_tangent;
                const float affine_value = apic ? affine_velocity(affine, particle, axis, sample.offset) : 0.0F;
                const float affine_delta = apic ? affine_velocity(affine_tangent, particle, axis, sample.offset) - affine_velocity(affine, particle, axis, position_delta) : 0.0F;
                const float velocity = component(velocities, axis)[particle] + affine_value;
                const float velocity_delta = component(velocity_tangent, axis)[particle] + affine_delta;
                const float gravity = axis == 0 ? gravity_x : axis == 1 ? gravity_y : gravity_z;
                const float acceleration = component(accelerations, axis)[particle] + gravity;
                const float acceleration_delta = component(acceleration_tangent, axis)[particle];
                mass_sum_tangent += mass_weight_tangent;
                momentum_sum_tangent += mass_weight_tangent * velocity + mass_weight * velocity_delta;
                acceleration_sum_tangent += mass_weight_tangent * acceleration + mass_weight * acceleration_delta;
            }
            const int axis = face < face_offset(configuration, 1) ? 0 : face < face_offset(configuration, 2) ? 1 : 2;
            const std::uint64_t local_face = face - face_offset(configuration, axis);
            const float mass = face_mass[face];
            if (mass == 0.0F) {
                component(old_velocity_tangent, axis)[local_face] = 0.0F;
                component(forced_velocity_tangent, axis)[local_face] = 0.0F;
                return;
            }
            component(old_velocity_tangent, axis)[local_face] = (momentum_sum_tangent - component(old_velocity, axis)[local_face] * mass_sum_tangent) / mass;
            component(forced_velocity_tangent, axis)[local_face] = (momentum_sum_tangent + configuration.time_step * acceleration_sum_tangent - component(forced_velocity, axis)[local_face] * mass_sum_tangent) / mass;
        }

        __global__ void p2g_vjp_kernel(const Configuration configuration, const std::uint32_t particle_count, const bool apic, const ConstVectorField positions, const ConstVectorField velocities, const ConstMatrixField affine, const ConstVectorField accelerations, const float* masses, const float gravity_x, const float gravity_y, const float gravity_z, const float* face_mass, const ConstStaggeredField old_velocity, const ConstStaggeredField forced_velocity, const ConstStaggeredAdjointField old_velocity_adjoint, const ConstStaggeredAdjointField forced_velocity_adjoint, const VectorAdjointField position_adjoint, const VectorAdjointField velocity_adjoint, const MatrixAdjointField affine_adjoint, const VectorAdjointField acceleration_adjoint, double* mass_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            double position_bar[3]{};
            double velocity_bar[3]{};
            double acceleration_bar[3]{};
            double affine_bar[9]{};
            double mass_bar = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                for (int slot = 0; slot < 27; ++slot) {
                    const Weight sample = weight(configuration, position, axis, slot);
                    if (!sample.valid) continue;
                    const float mass = face_mass[sample.face];
                    if (mass == 0.0F) continue;
                    const std::uint64_t local_face = sample.face - face_offset(configuration, axis);
                    const double old_bar = component(old_velocity_adjoint, axis)[local_face];
                    const double forced_bar = component(forced_velocity_adjoint, axis)[local_face];
                    const double momentum_bar = (old_bar + forced_bar) / mass;
                    const double acceleration_sum_bar = configuration.time_step * forced_bar / mass;
                    const double mass_sum_bar = -old_bar * component(old_velocity, axis)[local_face] / mass - forced_bar * component(forced_velocity, axis)[local_face] / mass;
                    const float velocity = component(velocities, axis)[particle] + (apic ? affine_velocity(affine, particle, axis, sample.offset) : 0.0F);
                    const float gravity = axis == 0 ? gravity_x : axis == 1 ? gravity_y : gravity_z;
                    const float acceleration = component(accelerations, axis)[particle] + gravity;
                    const double common = mass_sum_bar + velocity * momentum_bar + acceleration * acceleration_sum_bar;
                    mass_bar += sample.value * common;
                    const double weight_bar = masses[particle] * common;
                    position_bar[0] += weight_bar * sample.gradient.x;
                    position_bar[1] += weight_bar * sample.gradient.y;
                    position_bar[2] += weight_bar * sample.gradient.z;
                    const double velocity_value_bar = masses[particle] * sample.value * momentum_bar;
                    const double acceleration_value_bar = masses[particle] * sample.value * acceleration_sum_bar;
                    velocity_bar[axis] += velocity_value_bar;
                    acceleration_bar[axis] += acceleration_value_bar;
                    if (apic) {
                        affine_bar[axis * 3] += velocity_value_bar * sample.offset.x;
                        affine_bar[axis * 3 + 1] += velocity_value_bar * sample.offset.y;
                        affine_bar[axis * 3 + 2] += velocity_value_bar * sample.offset.z;
                        position_bar[0] -= velocity_value_bar * affine.values[axis * 3][particle];
                        position_bar[1] -= velocity_value_bar * affine.values[axis * 3 + 1][particle];
                        position_bar[2] -= velocity_value_bar * affine.values[axis * 3 + 2][particle];
                    }
                }
            }
            position_adjoint.x[particle] += position_bar[0];
            position_adjoint.y[particle] += position_bar[1];
            position_adjoint.z[particle] += position_bar[2];
            velocity_adjoint.x[particle] += velocity_bar[0];
            velocity_adjoint.y[particle] += velocity_bar[1];
            velocity_adjoint.z[particle] += velocity_bar[2];
            acceleration_adjoint.x[particle] += acceleration_bar[0];
            acceleration_adjoint.y[particle] += acceleration_bar[1];
            acceleration_adjoint.z[particle] += acceleration_bar[2];
            mass_adjoint[particle] += mass_bar;
            if (apic) for (int component_index = 0; component_index < 9; ++component_index) affine_adjoint.values[component_index][particle] += affine_bar[component_index];
        }

        __device__ bool solid_face(const Configuration configuration, const std::uint32_t* types, const int axis, const int x, const int y, const int z) {
            const int left_x = x - (axis == 0 ? 1 : 0);
            const int left_y = y - (axis == 1 ? 1 : 0);
            const int left_z = z - (axis == 2 ? 1 : 0);
            const std::uint32_t left = cell_type(types, configuration, left_x, left_y, left_z);
            const std::uint32_t right = cell_type(types, configuration, x, y, z);
            return left == 2u || right == 2u;
        }

        __device__ bool fluid_face(const Configuration configuration, const std::uint32_t* types, const int axis, const int x, const int y, const int z) {
            const int left_x = x - (axis == 0 ? 1 : 0);
            const int left_y = y - (axis == 1 ? 1 : 0);
            const int left_z = z - (axis == 2 ? 1 : 0);
            const std::uint32_t left = cell_type(types, configuration, left_x, left_y, left_z);
            const std::uint32_t right = cell_type(types, configuration, x, y, z);
            return !solid_face(configuration, types, axis, x, y, z) && (left == 1u || right == 1u);
        }

        __global__ void divergence_forward_kernel(const Configuration configuration, const std::uint32_t* types, const ConstStaggeredField solid_velocity, const ConstStaggeredField velocity, float* divergence) {
            const std::uint64_t cell = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cell >= cells(configuration)) return;
            if (types[cell] != 1u) {
                divergence[cell] = 0.0F;
                return;
            }
            int x;
            int y;
            int z;
            decode(cell, configuration.nx, configuration.ny, x, y, z);
            const std::uint64_t x_left = index3(x, y, z, configuration.nx + 1, configuration.ny);
            const std::uint64_t x_right = index3(x + 1, y, z, configuration.nx + 1, configuration.ny);
            const std::uint64_t y_bottom = index3(x, y, z, configuration.nx, configuration.ny + 1);
            const std::uint64_t y_top = index3(x, y + 1, z, configuration.nx, configuration.ny + 1);
            const std::uint64_t z_back = index3(x, y, z, configuration.nx, configuration.ny);
            const std::uint64_t z_front = index3(x, y, z + 1, configuration.nx, configuration.ny);
            const float left = cell_type(types, configuration, x - 1, y, z) == 2u ? solid_velocity.x[x_left] : velocity.x[x_left];
            const float right = cell_type(types, configuration, x + 1, y, z) == 2u ? solid_velocity.x[x_right] : velocity.x[x_right];
            const float bottom = cell_type(types, configuration, x, y - 1, z) == 2u ? solid_velocity.y[y_bottom] : velocity.y[y_bottom];
            const float top = cell_type(types, configuration, x, y + 1, z) == 2u ? solid_velocity.y[y_top] : velocity.y[y_top];
            const float back = cell_type(types, configuration, x, y, z - 1) == 2u ? solid_velocity.z[z_back] : velocity.z[z_back];
            const float front = cell_type(types, configuration, x, y, z + 1) == 2u ? solid_velocity.z[z_front] : velocity.z[z_front];
            divergence[cell] = (right - left + top - bottom + front - back) / configuration.cell_size;
        }

        __global__ void divergence_jvp_kernel(const Configuration configuration, const std::uint32_t* types, const ConstStaggeredField velocity_tangent, float* divergence_tangent) {
            const std::uint64_t cell = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cell >= cells(configuration)) return;
            if (types[cell] != 1u) {
                divergence_tangent[cell] = 0.0F;
                return;
            }
            int x;
            int y;
            int z;
            decode(cell, configuration.nx, configuration.ny, x, y, z);
            float value = 0.0F;
            if (cell_type(types, configuration, x - 1, y, z) != 2u) value -= velocity_tangent.x[index3(x, y, z, configuration.nx + 1, configuration.ny)];
            if (cell_type(types, configuration, x + 1, y, z) != 2u) value += velocity_tangent.x[index3(x + 1, y, z, configuration.nx + 1, configuration.ny)];
            if (cell_type(types, configuration, x, y - 1, z) != 2u) value -= velocity_tangent.y[index3(x, y, z, configuration.nx, configuration.ny + 1)];
            if (cell_type(types, configuration, x, y + 1, z) != 2u) value += velocity_tangent.y[index3(x, y + 1, z, configuration.nx, configuration.ny + 1)];
            if (cell_type(types, configuration, x, y, z - 1) != 2u) value -= velocity_tangent.z[index3(x, y, z, configuration.nx, configuration.ny)];
            if (cell_type(types, configuration, x, y, z + 1) != 2u) value += velocity_tangent.z[index3(x, y, z + 1, configuration.nx, configuration.ny)];
            divergence_tangent[cell] = value / configuration.cell_size;
        }

        __device__ int pressure_diagonal(const Configuration configuration, const std::uint32_t* types, const int x, const int y, const int z) {
            int diagonal = 0;
            diagonal += cell_type(types, configuration, x - 1, y, z) != 2u;
            diagonal += cell_type(types, configuration, x + 1, y, z) != 2u;
            diagonal += cell_type(types, configuration, x, y - 1, z) != 2u;
            diagonal += cell_type(types, configuration, x, y + 1, z) != 2u;
            diagonal += cell_type(types, configuration, x, y, z - 1) != 2u;
            diagonal += cell_type(types, configuration, x, y, z + 1) != 2u;
            return diagonal;
        }

        __device__ float pressure_value(const Configuration configuration, const std::uint32_t* types, const float* pressure, const int x, const int y, const int z) {
            if (cell_type(types, configuration, x, y, z) != 1u) return 0.0F;
            return pressure[index3(x, y, z, configuration.nx, configuration.ny)];
        }

        __global__ void pressure_iteration_kernel(const Configuration configuration, const std::uint32_t* types, const float* divergence, const float* previous, float* next) {
            const std::uint64_t cell = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cell >= cells(configuration)) return;
            if (types[cell] != 1u) {
                next[cell] = 0.0F;
                return;
            }
            if (configuration.pressure_anchor_enabled && cell == configuration.pressure_anchor) {
                next[cell] = 0.0F;
                return;
            }
            int x;
            int y;
            int z;
            decode(cell, configuration.nx, configuration.ny, x, y, z);
            const int diagonal = pressure_diagonal(configuration, types, x, y, z);
            float sum = 0.0F;
            sum += pressure_value(configuration, types, previous, x - 1, y, z);
            sum += pressure_value(configuration, types, previous, x + 1, y, z);
            sum += pressure_value(configuration, types, previous, x, y - 1, z);
            sum += pressure_value(configuration, types, previous, x, y + 1, z);
            sum += pressure_value(configuration, types, previous, x, y, z - 1);
            sum += pressure_value(configuration, types, previous, x, y, z + 1);
            next[cell] = (sum - configuration.cell_size * configuration.cell_size * divergence[cell] / configuration.time_step) / diagonal;
        }

        __global__ void project_kernel(const Configuration configuration, const std::uint32_t* types, const ConstStaggeredField solid_velocity, const ConstStaggeredField velocity, const float* pressure, const StaggeredField projected) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            const int axis = global_face < face_offset(configuration, 1) ? 0 : global_face < face_offset(configuration, 2) ? 1 : 2;
            const std::uint64_t face = global_face - face_offset(configuration, axis);
            int x;
            int y;
            int z;
            decode(face, extent(configuration, axis, 0), extent(configuration, axis, 1), x, y, z);
            if (solid_face(configuration, types, axis, x, y, z)) {
                component(projected, axis)[face] = component(solid_velocity, axis)[face];
                return;
            }
            const int left_x = x - (axis == 0 ? 1 : 0);
            const int left_y = y - (axis == 1 ? 1 : 0);
            const int left_z = z - (axis == 2 ? 1 : 0);
            const float left_pressure = pressure_value(configuration, types, pressure, left_x, left_y, left_z);
            const float right_pressure = pressure_value(configuration, types, pressure, x, y, z);
            component(projected, axis)[face] = component(velocity, axis)[face] - configuration.time_step * (right_pressure - left_pressure) / configuration.cell_size;
        }

        __global__ void project_jvp_kernel(const Configuration configuration, const std::uint32_t* types, const ConstStaggeredField velocity_tangent, const float* pressure_tangent, const StaggeredField projected_tangent) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            const int axis = global_face < face_offset(configuration, 1) ? 0 : global_face < face_offset(configuration, 2) ? 1 : 2;
            const std::uint64_t face = global_face - face_offset(configuration, axis);
            int x;
            int y;
            int z;
            decode(face, extent(configuration, axis, 0), extent(configuration, axis, 1), x, y, z);
            if (solid_face(configuration, types, axis, x, y, z)) {
                component(projected_tangent, axis)[face] = 0.0F;
                return;
            }
            const int left_x = x - (axis == 0 ? 1 : 0);
            const int left_y = y - (axis == 1 ? 1 : 0);
            const int left_z = z - (axis == 2 ? 1 : 0);
            const float left_pressure = pressure_value(configuration, types, pressure_tangent, left_x, left_y, left_z);
            const float right_pressure = pressure_value(configuration, types, pressure_tangent, x, y, z);
            component(projected_tangent, axis)[face] = component(velocity_tangent, axis)[face] - configuration.time_step * (right_pressure - left_pressure) / configuration.cell_size;
        }

        __global__ void pressure_final_adjoint_kernel(const Configuration configuration, const std::uint32_t* types, const ConstStaggeredAdjointField projected_adjoint, double* final_pressure_adjoint) {
            const std::uint64_t cell = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cell >= cells(configuration)) return;
            if (types[cell] != 1u) {
                final_pressure_adjoint[cell] = 0.0;
                return;
            }
            if (configuration.pressure_anchor_enabled && cell == configuration.pressure_anchor) {
                final_pressure_adjoint[cell] = 0.0;
                return;
            }
            int x;
            int y;
            int z;
            decode(cell, configuration.nx, configuration.ny, x, y, z);
            const double scale = configuration.time_step / configuration.cell_size;
            double value = 0.0;
            if (!solid_face(configuration, types, 0, x, y, z)) value -= scale * projected_adjoint.x[index3(x, y, z, configuration.nx + 1, configuration.ny)];
            if (!solid_face(configuration, types, 0, x + 1, y, z)) value += scale * projected_adjoint.x[index3(x + 1, y, z, configuration.nx + 1, configuration.ny)];
            if (!solid_face(configuration, types, 1, x, y, z)) value -= scale * projected_adjoint.y[index3(x, y, z, configuration.nx, configuration.ny + 1)];
            if (!solid_face(configuration, types, 1, x, y + 1, z)) value += scale * projected_adjoint.y[index3(x, y + 1, z, configuration.nx, configuration.ny + 1)];
            if (!solid_face(configuration, types, 2, x, y, z)) value -= scale * projected_adjoint.z[index3(x, y, z, configuration.nx, configuration.ny)];
            if (!solid_face(configuration, types, 2, x, y, z + 1)) value += scale * projected_adjoint.z[index3(x, y, z + 1, configuration.nx, configuration.ny)];
            final_pressure_adjoint[cell] = value;
        }

        __global__ void pressure_iteration_adjoint_kernel(const Configuration configuration, const std::uint32_t* types, const double* next_adjoint, double* previous_adjoint, double* divergence_adjoint) {
            const std::uint64_t cell = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cell >= cells(configuration)) return;
            if (types[cell] != 1u) {
                previous_adjoint[cell] = 0.0;
                return;
            }
            if (configuration.pressure_anchor_enabled && cell == configuration.pressure_anchor) {
                previous_adjoint[cell] = 0.0;
                return;
            }
            int x;
            int y;
            int z;
            decode(cell, configuration.nx, configuration.ny, x, y, z);
            double previous = 0.0;
            const int neighbor_x[6]{x - 1, x + 1, x, x, x, x};
            const int neighbor_y[6]{y, y, y - 1, y + 1, y, y};
            const int neighbor_z[6]{z, z, z, z, z - 1, z + 1};
            for (int neighbor = 0; neighbor < 6; ++neighbor) {
                if (cell_type(types, configuration, neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor]) != 1u) continue;
                const std::uint64_t neighbor_cell = index3(neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor], configuration.nx, configuration.ny);
                if (configuration.pressure_anchor_enabled && neighbor_cell == configuration.pressure_anchor) continue;
                const int diagonal = pressure_diagonal(configuration, types, neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor]);
                previous += next_adjoint[neighbor_cell] / diagonal;
            }
            previous_adjoint[cell] = previous;
            const int diagonal = pressure_diagonal(configuration, types, x, y, z);
            divergence_adjoint[cell] += -configuration.cell_size * configuration.cell_size * next_adjoint[cell] / (configuration.time_step * diagonal);
        }

        __global__ void velocity_adjoint_kernel(const Configuration configuration, const std::uint32_t* types, const ConstStaggeredAdjointField projected_adjoint, const double* divergence_adjoint, const StaggeredAdjointField velocity_adjoint) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            const int axis = global_face < face_offset(configuration, 1) ? 0 : global_face < face_offset(configuration, 2) ? 1 : 2;
            const std::uint64_t face = global_face - face_offset(configuration, axis);
            int x;
            int y;
            int z;
            decode(face, extent(configuration, axis, 0), extent(configuration, axis, 1), x, y, z);
            if (solid_face(configuration, types, axis, x, y, z)) {
                component(velocity_adjoint, axis)[face] = 0.0;
                return;
            }
            double value = component(projected_adjoint, axis)[face];
            const int left_x = x - (axis == 0 ? 1 : 0);
            const int left_y = y - (axis == 1 ? 1 : 0);
            const int left_z = z - (axis == 2 ? 1 : 0);
            if (cell_type(types, configuration, left_x, left_y, left_z) == 1u) value += divergence_adjoint[index3(left_x, left_y, left_z, configuration.nx, configuration.ny)] / configuration.cell_size;
            if (cell_type(types, configuration, x, y, z) == 1u) value -= divergence_adjoint[index3(x, y, z, configuration.nx, configuration.ny)] / configuration.cell_size;
            component(velocity_adjoint, axis)[face] += value;
        }

        __device__ void decode_global_face(const Configuration configuration, const std::uint64_t global_face, int& axis, std::uint64_t& face, int& x, int& y, int& z) {
            axis = global_face < face_offset(configuration, 1) ? 0 : global_face < face_offset(configuration, 2) ? 1 : 2;
            face = global_face - face_offset(configuration, axis);
            decode(face, extent(configuration, axis, 0), extent(configuration, axis, 1), x, y, z);
        }

        __device__ bool face_coordinate_valid(const Configuration configuration, const int axis, const int x, const int y, const int z) {
            return x >= 0 && y >= 0 && z >= 0 && x < extent(configuration, axis, 0) && y < extent(configuration, axis, 1) && z < extent(configuration, axis, 2);
        }

        __device__ std::uint64_t global_face_index(const Configuration configuration, const int axis, const int x, const int y, const int z) {
            return face_offset(configuration, axis) + index3(x, y, z, extent(configuration, axis, 0), extent(configuration, axis, 1));
        }

        __global__ void extrapolation_initial_kernel(const Configuration configuration, const std::uint32_t* types, const ConstStaggeredField solid_velocity, const ConstStaggeredField projected_velocity, std::uint32_t* valid, float* velocity) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            int axis;
            std::uint64_t face;
            int x;
            int y;
            int z;
            decode_global_face(configuration, global_face, axis, face, x, y, z);
            if (solid_face(configuration, types, axis, x, y, z)) {
                valid[global_face] = 1u;
                velocity[global_face] = component(solid_velocity, axis)[face];
                return;
            }
            valid[global_face] = fluid_face(configuration, types, axis, x, y, z) ? 1u : 0u;
            velocity[global_face] = valid[global_face] != 0u ? component(projected_velocity, axis)[face] : 0.0F;
        }

        __device__ int valid_neighbor_count(const Configuration configuration, const std::uint32_t* valid, const int axis, const int x, const int y, const int z) {
            const int neighbor_x[6]{x - 1, x + 1, x, x, x, x};
            const int neighbor_y[6]{y, y, y - 1, y + 1, y, y};
            const int neighbor_z[6]{z, z, z, z, z - 1, z + 1};
            int count = 0;
            for (int neighbor = 0; neighbor < 6; ++neighbor) if (face_coordinate_valid(configuration, axis, neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor])) count += valid[global_face_index(configuration, axis, neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor])] != 0u;
            return count;
        }

        __global__ void extrapolation_iteration_kernel(const Configuration configuration, const std::uint32_t* previous_valid, const float* previous_velocity, std::uint32_t* next_valid, float* next_velocity) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            if (previous_valid[global_face] != 0u) {
                next_valid[global_face] = 1u;
                next_velocity[global_face] = previous_velocity[global_face];
                return;
            }
            int axis;
            std::uint64_t face;
            int x;
            int y;
            int z;
            decode_global_face(configuration, global_face, axis, face, x, y, z);
            const int neighbor_x[6]{x - 1, x + 1, x, x, x, x};
            const int neighbor_y[6]{y, y, y - 1, y + 1, y, y};
            const int neighbor_z[6]{z, z, z, z, z - 1, z + 1};
            float sum = 0.0F;
            int count = 0;
            for (int neighbor = 0; neighbor < 6; ++neighbor) {
                if (!face_coordinate_valid(configuration, axis, neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor])) continue;
                const std::uint64_t neighbor_face = global_face_index(configuration, axis, neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor]);
                if (previous_valid[neighbor_face] == 0u) continue;
                sum += previous_velocity[neighbor_face];
                ++count;
            }
            next_valid[global_face] = count == 0 ? 0u : 1u;
            next_velocity[global_face] = count == 0 ? 0.0F : sum / count;
        }

        __global__ void copy_flat_to_staggered_kernel(const Configuration configuration, const float* flat, const StaggeredField staggered) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            int axis;
            std::uint64_t face;
            int x;
            int y;
            int z;
            decode_global_face(configuration, global_face, axis, face, x, y, z);
            component(staggered, axis)[face] = flat[global_face];
        }

        __global__ void extrapolation_initial_jvp_kernel(const Configuration configuration, const std::uint32_t* types, const ConstStaggeredField projected_tangent, float* velocity_tangent) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            int axis;
            std::uint64_t face;
            int x;
            int y;
            int z;
            decode_global_face(configuration, global_face, axis, face, x, y, z);
            velocity_tangent[global_face] = fluid_face(configuration, types, axis, x, y, z) ? component(projected_tangent, axis)[face] : 0.0F;
        }

        __global__ void extrapolation_iteration_jvp_kernel(const Configuration configuration, const std::uint32_t* previous_valid, const float* previous_tangent, float* next_tangent) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            if (previous_valid[global_face] != 0u) {
                next_tangent[global_face] = previous_tangent[global_face];
                return;
            }
            int axis;
            std::uint64_t face;
            int x;
            int y;
            int z;
            decode_global_face(configuration, global_face, axis, face, x, y, z);
            const int neighbor_x[6]{x - 1, x + 1, x, x, x, x};
            const int neighbor_y[6]{y, y, y - 1, y + 1, y, y};
            const int neighbor_z[6]{z, z, z, z, z - 1, z + 1};
            float sum = 0.0F;
            int count = 0;
            for (int neighbor = 0; neighbor < 6; ++neighbor) {
                if (!face_coordinate_valid(configuration, axis, neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor])) continue;
                const std::uint64_t neighbor_face = global_face_index(configuration, axis, neighbor_x[neighbor], neighbor_y[neighbor], neighbor_z[neighbor]);
                if (previous_valid[neighbor_face] == 0u) continue;
                sum += previous_tangent[neighbor_face];
                ++count;
            }
            next_tangent[global_face] = count == 0 ? 0.0F : sum / count;
        }

        __global__ void copy_staggered_adjoint_to_flat_kernel(const Configuration configuration, const ConstStaggeredAdjointField staggered, double* flat) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            int axis;
            std::uint64_t face;
            int x;
            int y;
            int z;
            decode_global_face(configuration, global_face, axis, face, x, y, z);
            flat[global_face] = component(staggered, axis)[face];
        }

        __global__ void extrapolation_iteration_vjp_kernel(const Configuration configuration, const std::uint32_t* previous_valid, const double* next_adjoint, double* previous_adjoint) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            int axis;
            std::uint64_t face;
            int x;
            int y;
            int z;
            decode_global_face(configuration, global_face, axis, face, x, y, z);
            double value = previous_valid[global_face] != 0u ? next_adjoint[global_face] : 0.0;
            if (previous_valid[global_face] != 0u) {
                const int target_x[6]{x - 1, x + 1, x, x, x, x};
                const int target_y[6]{y, y, y - 1, y + 1, y, y};
                const int target_z[6]{z, z, z, z, z - 1, z + 1};
                for (int target = 0; target < 6; ++target) {
                    if (!face_coordinate_valid(configuration, axis, target_x[target], target_y[target], target_z[target])) continue;
                    const std::uint64_t target_face = global_face_index(configuration, axis, target_x[target], target_y[target], target_z[target]);
                    if (previous_valid[target_face] != 0u) continue;
                    const int count = valid_neighbor_count(configuration, previous_valid, axis, target_x[target], target_y[target], target_z[target]);
                    if (count != 0) value += next_adjoint[target_face] / count;
                }
            }
            previous_adjoint[global_face] = value;
        }

        __global__ void extrapolation_initial_vjp_kernel(const Configuration configuration, const std::uint32_t* types, const double* initial_adjoint, const StaggeredAdjointField projected_adjoint) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            int axis;
            std::uint64_t face;
            int x;
            int y;
            int z;
            decode_global_face(configuration, global_face, axis, face, x, y, z);
            if (fluid_face(configuration, types, axis, x, y, z)) component(projected_adjoint, axis)[face] += initial_adjoint[global_face];
        }

        __device__ float sample_component(const Configuration configuration, const Vector position, const int axis, const ConstStaggeredField field) {
            float value = 0.0F;
            for (int slot = 0; slot < 27; ++slot) {
                const Weight sample = weight(configuration, position, axis, slot);
                if (sample.valid) value += sample.value * component(field, axis)[sample.face - face_offset(configuration, axis)];
            }
            return value;
        }

        __device__ float sample_component_tangent(const Configuration configuration, const Vector position, const Vector position_tangent, const int axis, const ConstStaggeredField field, const ConstStaggeredField field_tangent) {
            float value = 0.0F;
            for (int slot = 0; slot < 27; ++slot) {
                const Weight sample = weight(configuration, position, axis, slot);
                if (sample.valid) value += dot(sample.gradient, position_tangent) * component(field, axis)[sample.face - face_offset(configuration, axis)] + sample.value * component(field_tangent, axis)[sample.face - face_offset(configuration, axis)];
            }
            return value;
        }

        __global__ void g2p_pic_flip_forward_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstVectorField particle_velocity, const ConstStaggeredField old_grid_velocity, const ConstStaggeredField new_grid_velocity, const float* blend, const VectorField velocity) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            Vector output{};
            for (int axis = 0; axis < 3; ++axis) {
                const float pic = sample_component(configuration, position, axis, new_grid_velocity);
                const float delta = pic - sample_component(configuration, position, axis, old_grid_velocity);
                const float flip = component(particle_velocity, axis)[particle] + delta;
                component(output, axis) = (1.0F - blend[0]) * pic + blend[0] * flip;
            }
            store(velocity, particle, output);
        }

        __global__ void g2p_pic_flip_jvp_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstVectorField particle_velocity, const ConstStaggeredField old_grid_velocity, const ConstStaggeredField new_grid_velocity, const float* blend, const ConstVectorField position_tangent, const ConstVectorField particle_velocity_tangent, const ConstStaggeredField old_grid_velocity_tangent, const ConstStaggeredField new_grid_velocity_tangent, const float* blend_tangent, const VectorField velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            const Vector position_delta = load(position_tangent, particle);
            Vector output{};
            for (int axis = 0; axis < 3; ++axis) {
                const float pic = sample_component(configuration, position, axis, new_grid_velocity);
                const float old = sample_component(configuration, position, axis, old_grid_velocity);
                const float pic_delta = sample_component_tangent(configuration, position, position_delta, axis, new_grid_velocity, new_grid_velocity_tangent);
                const float old_delta = sample_component_tangent(configuration, position, position_delta, axis, old_grid_velocity, old_grid_velocity_tangent);
                const float flip = component(particle_velocity, axis)[particle] + pic - old;
                const float flip_delta = component(particle_velocity_tangent, axis)[particle] + pic_delta - old_delta;
                component(output, axis) = (1.0F - blend[0]) * pic_delta + blend[0] * flip_delta + blend_tangent[0] * (flip - pic);
            }
            store(velocity_tangent, particle, output);
        }

        __global__ void g2p_pic_flip_particle_vjp_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstVectorField particle_velocity, const ConstStaggeredField old_grid_velocity, const ConstStaggeredField new_grid_velocity, const float* blend, const ConstVectorAdjointField output_adjoint, const VectorAdjointField position_adjoint, const VectorAdjointField particle_velocity_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            double position_bar[3]{};
            for (int axis = 0; axis < 3; ++axis) {
                const double output_bar = component(output_adjoint, axis)[particle];
                component(particle_velocity_adjoint, axis)[particle] += blend[0] * output_bar;
                for (int slot = 0; slot < 27; ++slot) {
                    const Weight sample = weight(configuration, position, axis, slot);
                    if (!sample.valid) continue;
                    const std::uint64_t face = sample.face - face_offset(configuration, axis);
                    const double weight_bar = output_bar * (component(new_grid_velocity, axis)[face] - blend[0] * component(old_grid_velocity, axis)[face]);
                    position_bar[0] += weight_bar * sample.gradient.x;
                    position_bar[1] += weight_bar * sample.gradient.y;
                    position_bar[2] += weight_bar * sample.gradient.z;
                }
            }
            position_adjoint.x[particle] += position_bar[0];
            position_adjoint.y[particle] += position_bar[1];
            position_adjoint.z[particle] += position_bar[2];
        }

        __global__ void g2p_pic_flip_grid_vjp_kernel(const Configuration configuration, const std::size_t contribution_count, const float* blend, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const ConstVectorField positions, const ConstVectorAdjointField output_adjoint, const StaggeredAdjointField old_grid_adjoint, const StaggeredAdjointField new_grid_adjoint) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            const std::size_t begin = lower_bound(sorted_keys, contribution_count, global_face);
            double old_bar = 0.0;
            double new_bar = 0.0;
            for (std::size_t entry = begin; entry < contribution_count && sorted_keys[entry] == global_face; ++entry) {
                const std::uint32_t id = sorted_ids[entry];
                const std::uint32_t particle = id / 81u;
                const int local = id % 81u;
                const int axis = local / 27;
                const Weight sample = weight(configuration, load(positions, particle), axis, local % 27);
                const double output_bar = component(output_adjoint, axis)[particle];
                old_bar -= blend[0] * sample.value * output_bar;
                new_bar += sample.value * output_bar;
            }
            const int axis = global_face < face_offset(configuration, 1) ? 0 : global_face < face_offset(configuration, 2) ? 1 : 2;
            const std::uint64_t face = global_face - face_offset(configuration, axis);
            component(old_grid_adjoint, axis)[face] += old_bar;
            component(new_grid_adjoint, axis)[face] += new_bar;
        }

        __global__ void g2p_pic_flip_blend_values_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstVectorField particle_velocity, const ConstStaggeredField old_grid_velocity, const ConstStaggeredField new_grid_velocity, const ConstVectorAdjointField output_adjoint, double* values) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            double value = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const float pic = sample_component(configuration, position, axis, new_grid_velocity);
                const float flip = component(particle_velocity, axis)[particle] + pic - sample_component(configuration, position, axis, old_grid_velocity);
                value += component(output_adjoint, axis)[particle] * (flip - pic);
            }
            values[particle] = value;
        }

        __global__ void accumulate_scalar_kernel(const double* source, double* destination) {
            if (blockIdx.x == 0u && threadIdx.x == 0u) destination[0] += source[0];
        }

        __global__ void g2p_apic_forward_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstStaggeredField grid_velocity, const VectorField velocity, const MatrixField affine) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            const float factor = 4.0F / (configuration.cell_size * configuration.cell_size);
            for (int axis = 0; axis < 3; ++axis) {
                float value = 0.0F;
                float affine_value[3]{};
                for (int slot = 0; slot < 27; ++slot) {
                    const Weight sample = weight(configuration, position, axis, slot);
                    if (!sample.valid) continue;
                    const float grid_value = component(grid_velocity, axis)[sample.face - face_offset(configuration, axis)];
                    value += sample.value * grid_value;
                    affine_value[0] += factor * sample.value * grid_value * sample.offset.x;
                    affine_value[1] += factor * sample.value * grid_value * sample.offset.y;
                    affine_value[2] += factor * sample.value * grid_value * sample.offset.z;
                }
                component(velocity, axis)[particle] = value;
                affine.values[axis * 3][particle] = affine_value[0];
                affine.values[axis * 3 + 1][particle] = affine_value[1];
                affine.values[axis * 3 + 2][particle] = affine_value[2];
            }
        }

        __global__ void g2p_apic_jvp_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstStaggeredField grid_velocity, const ConstVectorField position_tangent, const ConstStaggeredField grid_velocity_tangent, const VectorField velocity_tangent, const MatrixField affine_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            const Vector position_delta = load(position_tangent, particle);
            const float factor = 4.0F / (configuration.cell_size * configuration.cell_size);
            for (int axis = 0; axis < 3; ++axis) {
                float value_delta = 0.0F;
                float affine_delta[3]{};
                for (int slot = 0; slot < 27; ++slot) {
                    const Weight sample = weight(configuration, position, axis, slot);
                    if (!sample.valid) continue;
                    const std::uint64_t face = sample.face - face_offset(configuration, axis);
                    const float grid_value = component(grid_velocity, axis)[face];
                    const float grid_delta = component(grid_velocity_tangent, axis)[face];
                    const float weight_delta = dot(sample.gradient, position_delta);
                    const float weighted_delta = weight_delta * grid_value + sample.value * grid_delta;
                    value_delta += weighted_delta;
                    affine_delta[0] += factor * (weighted_delta * sample.offset.x - sample.value * grid_value * position_delta.x);
                    affine_delta[1] += factor * (weighted_delta * sample.offset.y - sample.value * grid_value * position_delta.y);
                    affine_delta[2] += factor * (weighted_delta * sample.offset.z - sample.value * grid_value * position_delta.z);
                }
                component(velocity_tangent, axis)[particle] = value_delta;
                affine_tangent.values[axis * 3][particle] = affine_delta[0];
                affine_tangent.values[axis * 3 + 1][particle] = affine_delta[1];
                affine_tangent.values[axis * 3 + 2][particle] = affine_delta[2];
            }
        }

        __global__ void g2p_apic_particle_vjp_kernel(const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstStaggeredField grid_velocity, const ConstVectorAdjointField velocity_adjoint, const ConstMatrixAdjointField affine_adjoint, const VectorAdjointField position_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            const double factor = 4.0 / (configuration.cell_size * configuration.cell_size);
            double position_bar[3]{};
            for (int axis = 0; axis < 3; ++axis) {
                const double velocity_bar = component(velocity_adjoint, axis)[particle];
                const double matrix_bar[3]{affine_adjoint.values[axis * 3][particle], affine_adjoint.values[axis * 3 + 1][particle], affine_adjoint.values[axis * 3 + 2][particle]};
                for (int slot = 0; slot < 27; ++slot) {
                    const Weight sample = weight(configuration, position, axis, slot);
                    if (!sample.valid) continue;
                    const float grid_value = component(grid_velocity, axis)[sample.face - face_offset(configuration, axis)];
                    const double weight_bar = grid_value * (velocity_bar + factor * (matrix_bar[0] * sample.offset.x + matrix_bar[1] * sample.offset.y + matrix_bar[2] * sample.offset.z));
                    position_bar[0] += weight_bar * sample.gradient.x - factor * sample.value * grid_value * matrix_bar[0];
                    position_bar[1] += weight_bar * sample.gradient.y - factor * sample.value * grid_value * matrix_bar[1];
                    position_bar[2] += weight_bar * sample.gradient.z - factor * sample.value * grid_value * matrix_bar[2];
                }
            }
            position_adjoint.x[particle] += position_bar[0];
            position_adjoint.y[particle] += position_bar[1];
            position_adjoint.z[particle] += position_bar[2];
        }

        __global__ void g2p_apic_grid_vjp_kernel(const Configuration configuration, const std::size_t contribution_count, const ConstVectorField positions, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const ConstVectorAdjointField velocity_adjoint, const ConstMatrixAdjointField affine_adjoint, const StaggeredAdjointField grid_velocity_adjoint) {
            const std::uint64_t global_face = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (global_face >= total_faces(configuration)) return;
            const std::size_t begin = lower_bound(sorted_keys, contribution_count, global_face);
            const double factor = 4.0 / (configuration.cell_size * configuration.cell_size);
            double value = 0.0;
            for (std::size_t entry = begin; entry < contribution_count && sorted_keys[entry] == global_face; ++entry) {
                const std::uint32_t id = sorted_ids[entry];
                const std::uint32_t particle = id / 81u;
                const int local = id % 81u;
                const int axis = local / 27;
                const Weight sample = weight(configuration, load(positions, particle), axis, local % 27);
                const double matrix_term = affine_adjoint.values[axis * 3][particle] * sample.offset.x + affine_adjoint.values[axis * 3 + 1][particle] * sample.offset.y + affine_adjoint.values[axis * 3 + 2][particle] * sample.offset.z;
                value += sample.value * (component(velocity_adjoint, axis)[particle] + factor * matrix_term);
            }
            const int axis = global_face < face_offset(configuration, 1) ? 0 : global_face < face_offset(configuration, 2) ? 1 : 2;
            component(grid_velocity_adjoint, axis)[global_face - face_offset(configuration, axis)] += value;
        }

        __global__ void advect_forward_kernel(const Configuration configuration, const std::uint32_t particle_count, const float particle_radius, const std::uint32_t* types, const ConstStaggeredField solid_velocity, const ConstVectorField positions, const ConstVectorField velocities, std::uint32_t* collision_masks, const VectorField unconstrained_positions, const VectorField next_positions, const VectorField next_velocities) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const Vector position = load(positions, particle);
            const Vector velocity = load(velocities, particle);
            const Vector unconstrained = position + configuration.time_step * velocity;
            store(unconstrained_positions, particle, unconstrained);
            const Vector minimum{configuration.origin_x + particle_radius, configuration.origin_y + particle_radius, configuration.origin_z + particle_radius};
            const Vector maximum{
                configuration.origin_x + configuration.nx * configuration.cell_size - particle_radius,
                configuration.origin_y + configuration.ny * configuration.cell_size - particle_radius,
                configuration.origin_z + configuration.nz * configuration.cell_size - particle_radius,
            };
            Vector constrained{
                .x = fminf(maximum.x, fmaxf(minimum.x, unconstrained.x)),
                .y = fminf(maximum.y, fmaxf(minimum.y, unconstrained.y)),
                .z = fminf(maximum.z, fmaxf(minimum.z, unconstrained.z)),
            };
            std::uint32_t mask = 0u;
            if (constrained.x != unconstrained.x) mask |= 1u;
            if (constrained.y != unconstrained.y) mask |= 2u;
            if (constrained.z != unconstrained.z) mask |= 4u;
            const int cell_x = static_cast<int>(floorf((constrained.x - configuration.origin_x) / configuration.cell_size));
            const int cell_y = static_cast<int>(floorf((constrained.y - configuration.origin_y) / configuration.cell_size));
            const int cell_z = static_cast<int>(floorf((constrained.z - configuration.origin_z) / configuration.cell_size));
            if (cell_type(types, configuration, cell_x, cell_y, cell_z) == 2u) mask |= 8u;
            Vector constrained_velocity = velocity;
            if ((mask & 8u) != 0u) {
                constrained = position;
                constrained_velocity = {
                    .x = 0.5F * (solid_velocity.x[index3(cell_x, cell_y, cell_z, configuration.nx + 1, configuration.ny)] + solid_velocity.x[index3(cell_x + 1, cell_y, cell_z, configuration.nx + 1, configuration.ny)]),
                    .y = 0.5F * (solid_velocity.y[index3(cell_x, cell_y, cell_z, configuration.nx, configuration.ny + 1)] + solid_velocity.y[index3(cell_x, cell_y + 1, cell_z, configuration.nx, configuration.ny + 1)]),
                    .z = 0.5F * (solid_velocity.z[index3(cell_x, cell_y, cell_z, configuration.nx, configuration.ny)] + solid_velocity.z[index3(cell_x, cell_y, cell_z + 1, configuration.nx, configuration.ny)]),
                };
            } else {
                if ((mask & 1u) != 0u) constrained_velocity.x = solid_velocity.x[index3(unconstrained.x < minimum.x ? 0 : configuration.nx, cell_y, cell_z, configuration.nx + 1, configuration.ny)];
                if ((mask & 2u) != 0u) constrained_velocity.y = solid_velocity.y[index3(cell_x, unconstrained.y < minimum.y ? 0 : configuration.ny, cell_z, configuration.nx, configuration.ny + 1)];
                if ((mask & 4u) != 0u) constrained_velocity.z = solid_velocity.z[index3(cell_x, cell_y, unconstrained.z < minimum.z ? 0 : configuration.nz, configuration.nx, configuration.ny)];
            }
            collision_masks[particle] = mask;
            store(next_positions, particle, constrained);
            store(next_velocities, particle, constrained_velocity);
        }

        __global__ void advect_jvp_kernel(const Configuration configuration, const std::uint32_t particle_count, const std::uint32_t* collision_masks, const ConstVectorField position_tangent, const ConstVectorField velocity_tangent, const VectorField next_position_tangent, const VectorField next_velocity_tangent) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const std::uint32_t mask = collision_masks[particle];
            const Vector position_delta = load(position_tangent, particle);
            const Vector velocity_delta = load(velocity_tangent, particle);
            if ((mask & 8u) != 0u) {
                store(next_position_tangent, particle, position_delta);
                store(next_velocity_tangent, particle, {});
                return;
            }
            Vector next_position_delta = position_delta + configuration.time_step * velocity_delta;
            Vector next_velocity_delta = velocity_delta;
            if ((mask & 1u) != 0u) {
                next_position_delta.x = 0.0F;
                next_velocity_delta.x = 0.0F;
            }
            if ((mask & 2u) != 0u) {
                next_position_delta.y = 0.0F;
                next_velocity_delta.y = 0.0F;
            }
            if ((mask & 4u) != 0u) {
                next_position_delta.z = 0.0F;
                next_velocity_delta.z = 0.0F;
            }
            store(next_position_tangent, particle, next_position_delta);
            store(next_velocity_tangent, particle, next_velocity_delta);
        }

        __global__ void advect_vjp_kernel(const Configuration configuration, const std::uint32_t particle_count, const std::uint32_t* collision_masks, const ConstVectorAdjointField next_position_adjoint, const ConstVectorAdjointField next_velocity_adjoint, const VectorAdjointField position_adjoint, const VectorAdjointField velocity_adjoint) {
            const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const std::uint32_t mask = collision_masks[particle];
            if ((mask & 8u) != 0u) {
                position_adjoint.x[particle] += next_position_adjoint.x[particle];
                position_adjoint.y[particle] += next_position_adjoint.y[particle];
                position_adjoint.z[particle] += next_position_adjoint.z[particle];
                return;
            }
            for (int axis = 0; axis < 3; ++axis) {
                if ((mask & (1u << axis)) != 0u) continue;
                component(position_adjoint, axis)[particle] += component(next_position_adjoint, axis)[particle];
                component(velocity_adjoint, axis)[particle] += configuration.time_step * component(next_position_adjoint, axis)[particle] + component(next_velocity_adjoint, axis)[particle];
            }
        }

        __global__ void accumulate_double_kernel(const std::size_t count, const double* source, double* destination) {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += source[index];
        }

    } // namespace

    std::size_t sort_scratch_bytes(const std::size_t contribution_count) {
        std::size_t bytes = 0u;
        check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const std::uint64_t*>(nullptr), static_cast<std::uint64_t*>(nullptr), static_cast<const std::uint32_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), contribution_count), "query P2G radix sort");
        return bytes;
    }

    std::size_t reduction_scratch_bytes(const std::size_t value_count) {
        std::size_t bytes = 0u;
        check(cub::DeviceReduce::Sum(nullptr, bytes, static_cast<const double*>(nullptr), static_cast<double*>(nullptr), value_count), "query deterministic reduction");
        return bytes;
    }

    void sort_contributions(const cudaStream_t stream, void* scratch, const std::size_t scratch_bytes, const std::uint64_t* keys, std::uint64_t* sorted_keys, const std::uint32_t* ids, std::uint32_t* sorted_ids, const std::size_t count) {
        std::size_t bytes = scratch_bytes;
        check(cub::DeviceRadixSort::SortPairs(scratch, bytes, keys, sorted_keys, ids, sorted_ids, count, 0, 64, stream), "sort P2G contributions");
    }

    void cell_topology(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, std::uint64_t* keys, std::uint32_t* ids) {
        cell_topology_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, positions, keys, ids);
        check_launch("build marker topology");
    }

    void marker_forward(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const std::uint64_t* sorted_cell_keys, const std::uint32_t* domain_cell_types, std::uint32_t* cell_types) {
        marker_kernel<<<blocks(cells(configuration)), block_size, 0, stream>>>(configuration, particle_count, sorted_cell_keys, domain_cell_types, cell_types);
        check_launch("mark fluid cells");
    }

    void p2g_topology(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, std::uint64_t* keys, std::uint32_t* ids) {
        const std::size_t count = static_cast<std::size_t>(particle_count) * 81u;
        p2g_topology_kernel<<<blocks(count), block_size, 0, stream>>>(configuration, particle_count, positions, keys, ids);
        check_launch("build P2G topology");
    }

    void p2g_forward(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const bool apic, const ConstVectorField positions, const ConstVectorField velocities, const ConstMatrixField affine, const ConstVectorField accelerations, const float* masses, const float gravity_x, const float gravity_y, const float gravity_z, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, float* face_mass, const StaggeredField old_velocity, const StaggeredField forced_velocity) {
        const std::size_t contributions = static_cast<std::size_t>(particle_count) * 81u;
        p2g_forward_kernel<<<blocks(total_faces(configuration)), block_size, 0, stream>>>(configuration, contributions, apic, positions, velocities, affine, accelerations, masses, gravity_x, gravity_y, gravity_z, sorted_keys, sorted_ids, face_mass, old_velocity, forced_velocity);
        check_launch("P2G forward");
    }

    void p2g_jvp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const bool apic, const ConstVectorField positions, const ConstVectorField velocities, const ConstMatrixField affine, const ConstVectorField accelerations, const float* masses, const float gravity_x, const float gravity_y, const float gravity_z, const ConstVectorField position_tangent, const ConstVectorField velocity_tangent, const ConstMatrixField affine_tangent, const ConstVectorField acceleration_tangent, const float* mass_tangent, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const float* face_mass, const ConstStaggeredField old_velocity, const ConstStaggeredField forced_velocity, const StaggeredField old_velocity_tangent, const StaggeredField forced_velocity_tangent) {
        const std::size_t contributions = static_cast<std::size_t>(particle_count) * 81u;
        p2g_jvp_kernel<<<blocks(total_faces(configuration)), block_size, 0, stream>>>(configuration, contributions, apic, positions, velocities, affine, accelerations, masses, gravity_x, gravity_y, gravity_z, position_tangent, velocity_tangent, affine_tangent, acceleration_tangent, mass_tangent, sorted_keys, sorted_ids, face_mass, old_velocity, forced_velocity, old_velocity_tangent, forced_velocity_tangent);
        check_launch("P2G JVP");
    }

    void p2g_vjp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const bool apic, const ConstVectorField positions, const ConstVectorField velocities, const ConstMatrixField affine, const ConstVectorField accelerations, const float* masses, const float gravity_x, const float gravity_y, const float gravity_z, const std::uint64_t*, const std::uint32_t*, const float* face_mass, const ConstStaggeredField old_velocity, const ConstStaggeredField forced_velocity, const ConstStaggeredAdjointField old_velocity_adjoint, const ConstStaggeredAdjointField forced_velocity_adjoint, const VectorAdjointField position_adjoint, const VectorAdjointField velocity_adjoint, const MatrixAdjointField affine_adjoint, const VectorAdjointField acceleration_adjoint, double* mass_adjoint) {
        p2g_vjp_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, apic, positions, velocities, affine, accelerations, masses, gravity_x, gravity_y, gravity_z, face_mass, old_velocity, forced_velocity, old_velocity_adjoint, forced_velocity_adjoint, position_adjoint, velocity_adjoint, affine_adjoint, acceleration_adjoint, mass_adjoint);
        check_launch("P2G VJP");
    }

    void projection_forward(const cudaStream_t stream, const Configuration configuration, const std::uint32_t* cell_types, const ConstStaggeredField solid_velocity, const ConstStaggeredField velocity, float* divergence, float* pressure_history, const StaggeredField projected_velocity) {
        const std::size_t cell_total = cells(configuration);
        divergence_forward_kernel<<<blocks(cell_total), block_size, 0, stream>>>(configuration, cell_types, solid_velocity, velocity, divergence);
        check_launch("projection divergence forward");
        check(cudaMemsetAsync(pressure_history, 0, cell_total * sizeof(float), stream), "initialize pressure");
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            const float* previous = pressure_history + static_cast<std::size_t>(iteration) * cell_total;
            float* next = pressure_history + static_cast<std::size_t>(iteration + 1u) * cell_total;
            pressure_iteration_kernel<<<blocks(cell_total), block_size, 0, stream>>>(configuration, cell_types, divergence, previous, next);
            check_launch("pressure Jacobi forward");
        }
        const float* pressure = pressure_history + static_cast<std::size_t>(configuration.pressure_iterations) * cell_total;
        project_kernel<<<blocks(total_faces(configuration)), block_size, 0, stream>>>(configuration, cell_types, solid_velocity, velocity, pressure, projected_velocity);
        check_launch("pressure gradient forward");
    }

    void projection_jvp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t* cell_types, const ConstStaggeredField velocity_tangent, float* divergence_tangent, float* pressure_tangent_history, const StaggeredField projected_velocity_tangent) {
        const std::size_t cell_total = cells(configuration);
        divergence_jvp_kernel<<<blocks(cell_total), block_size, 0, stream>>>(configuration, cell_types, velocity_tangent, divergence_tangent);
        check_launch("projection divergence JVP");
        check(cudaMemsetAsync(pressure_tangent_history, 0, cell_total * sizeof(float), stream), "initialize pressure tangent");
        for (std::uint32_t iteration = 0u; iteration < configuration.pressure_iterations; ++iteration) {
            const float* previous = pressure_tangent_history + static_cast<std::size_t>(iteration) * cell_total;
            float* next = pressure_tangent_history + static_cast<std::size_t>(iteration + 1u) * cell_total;
            pressure_iteration_kernel<<<blocks(cell_total), block_size, 0, stream>>>(configuration, cell_types, divergence_tangent, previous, next);
            check_launch("pressure Jacobi JVP");
        }
        const float* pressure = pressure_tangent_history + static_cast<std::size_t>(configuration.pressure_iterations) * cell_total;
        project_jvp_kernel<<<blocks(total_faces(configuration)), block_size, 0, stream>>>(configuration, cell_types, velocity_tangent, pressure, projected_velocity_tangent);
        check_launch("pressure gradient JVP");
    }

    void projection_vjp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t* cell_types, const ConstStaggeredAdjointField projected_velocity_adjoint, double* divergence_adjoint, double* pressure_adjoint_history, const StaggeredAdjointField velocity_adjoint) {
        const std::size_t cell_total = cells(configuration);
        check(cudaMemsetAsync(divergence_adjoint, 0, cell_total * sizeof(double), stream), "initialize divergence adjoint");
        check(cudaMemsetAsync(pressure_adjoint_history, 0, cell_total * (configuration.pressure_iterations + 1u) * sizeof(double), stream), "initialize pressure adjoint history");
        double* final_pressure_adjoint = pressure_adjoint_history + static_cast<std::size_t>(configuration.pressure_iterations) * cell_total;
        pressure_final_adjoint_kernel<<<blocks(cell_total), block_size, 0, stream>>>(configuration, cell_types, projected_velocity_adjoint, final_pressure_adjoint);
        check_launch("pressure gradient VJP");
        for (std::uint32_t iteration = configuration.pressure_iterations; iteration-- > 0u;) {
            const double* next = pressure_adjoint_history + static_cast<std::size_t>(iteration + 1u) * cell_total;
            double* previous = pressure_adjoint_history + static_cast<std::size_t>(iteration) * cell_total;
            pressure_iteration_adjoint_kernel<<<blocks(cell_total), block_size, 0, stream>>>(configuration, cell_types, next, previous, divergence_adjoint);
            check_launch("pressure Jacobi VJP");
        }
        velocity_adjoint_kernel<<<blocks(total_faces(configuration)), block_size, 0, stream>>>(configuration, cell_types, projected_velocity_adjoint, divergence_adjoint, velocity_adjoint);
        check_launch("projection divergence VJP");
    }

    void extrapolation_forward(const cudaStream_t stream, const Configuration configuration, const std::uint32_t* cell_types, const ConstStaggeredField solid_velocity, const ConstStaggeredField projected_velocity, std::uint32_t* valid_history, float* velocity_history, const StaggeredField extrapolated_velocity) {
        const std::size_t faces = total_faces(configuration);
        extrapolation_initial_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, cell_types, solid_velocity, projected_velocity, valid_history, velocity_history);
        check_launch("extrapolation initialize");
        for (std::uint32_t iteration = 0u; iteration < configuration.extrapolation_iterations; ++iteration) {
            const std::uint32_t* previous_valid = valid_history + static_cast<std::size_t>(iteration) * faces;
            const float* previous_velocity = velocity_history + static_cast<std::size_t>(iteration) * faces;
            std::uint32_t* next_valid = valid_history + static_cast<std::size_t>(iteration + 1u) * faces;
            float* next_velocity = velocity_history + static_cast<std::size_t>(iteration + 1u) * faces;
            extrapolation_iteration_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, previous_valid, previous_velocity, next_valid, next_velocity);
            check_launch("extrapolation iteration");
        }
        copy_flat_to_staggered_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, velocity_history + static_cast<std::size_t>(configuration.extrapolation_iterations) * faces, extrapolated_velocity);
        check_launch("extrapolation output");
    }

    void extrapolation_jvp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t* cell_types, const std::uint32_t* valid_history, const ConstStaggeredField projected_velocity_tangent, float* velocity_tangent_history, const StaggeredField extrapolated_velocity_tangent) {
        const std::size_t faces = total_faces(configuration);
        extrapolation_initial_jvp_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, cell_types, projected_velocity_tangent, velocity_tangent_history);
        check_launch("extrapolation JVP initialize");
        for (std::uint32_t iteration = 0u; iteration < configuration.extrapolation_iterations; ++iteration) {
            const std::uint32_t* previous_valid = valid_history + static_cast<std::size_t>(iteration) * faces;
            const float* previous_tangent = velocity_tangent_history + static_cast<std::size_t>(iteration) * faces;
            float* next_tangent = velocity_tangent_history + static_cast<std::size_t>(iteration + 1u) * faces;
            extrapolation_iteration_jvp_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, previous_valid, previous_tangent, next_tangent);
            check_launch("extrapolation JVP iteration");
        }
        copy_flat_to_staggered_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, velocity_tangent_history + static_cast<std::size_t>(configuration.extrapolation_iterations) * faces, extrapolated_velocity_tangent);
        check_launch("extrapolation JVP output");
    }

    void extrapolation_vjp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t* cell_types, const std::uint32_t* valid_history, const ConstStaggeredAdjointField extrapolated_velocity_adjoint, double* velocity_adjoint_history, const StaggeredAdjointField projected_velocity_adjoint) {
        const std::size_t faces = total_faces(configuration);
        check(cudaMemsetAsync(velocity_adjoint_history, 0, faces * (configuration.extrapolation_iterations + 1u) * sizeof(double), stream), "initialize extrapolation adjoint history");
        double* final_adjoint = velocity_adjoint_history + static_cast<std::size_t>(configuration.extrapolation_iterations) * faces;
        copy_staggered_adjoint_to_flat_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, extrapolated_velocity_adjoint, final_adjoint);
        check_launch("extrapolation VJP input");
        for (std::uint32_t iteration = configuration.extrapolation_iterations; iteration-- > 0u;) {
            const std::uint32_t* previous_valid = valid_history + static_cast<std::size_t>(iteration) * faces;
            const double* next_adjoint = velocity_adjoint_history + static_cast<std::size_t>(iteration + 1u) * faces;
            double* previous_adjoint = velocity_adjoint_history + static_cast<std::size_t>(iteration) * faces;
            extrapolation_iteration_vjp_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, previous_valid, next_adjoint, previous_adjoint);
            check_launch("extrapolation VJP iteration");
        }
        extrapolation_initial_vjp_kernel<<<blocks(faces), block_size, 0, stream>>>(configuration, cell_types, velocity_adjoint_history, projected_velocity_adjoint);
        check_launch("extrapolation VJP output");
    }

    void g2p_pic_flip_forward(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstVectorField particle_velocity, const ConstStaggeredField old_grid_velocity, const ConstStaggeredField new_grid_velocity, const float* blend, const VectorField velocity) {
        g2p_pic_flip_forward_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, positions, particle_velocity, old_grid_velocity, new_grid_velocity, blend, velocity);
        check_launch("PIC FLIP G2P forward");
    }

    void g2p_pic_flip_jvp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstVectorField particle_velocity, const ConstStaggeredField old_grid_velocity, const ConstStaggeredField new_grid_velocity, const float* blend, const ConstVectorField position_tangent, const ConstVectorField particle_velocity_tangent, const ConstStaggeredField old_grid_velocity_tangent, const ConstStaggeredField new_grid_velocity_tangent, const float* blend_tangent, const VectorField velocity_tangent) {
        g2p_pic_flip_jvp_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, positions, particle_velocity, old_grid_velocity, new_grid_velocity, blend, position_tangent, particle_velocity_tangent, old_grid_velocity_tangent, new_grid_velocity_tangent, blend_tangent, velocity_tangent);
        check_launch("PIC FLIP G2P JVP");
    }

    void g2p_pic_flip_vjp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstVectorField particle_velocity, const ConstStaggeredField old_grid_velocity, const ConstStaggeredField new_grid_velocity, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const float* blend, const ConstVectorAdjointField velocity_adjoint, const VectorAdjointField position_adjoint, const VectorAdjointField particle_velocity_adjoint, const StaggeredAdjointField old_grid_velocity_adjoint, const StaggeredAdjointField new_grid_velocity_adjoint, double* reduction_values, void* reduction_scratch, const std::size_t reduction_scratch_bytes, double* reduction_result, double* blend_adjoint) {
        const std::size_t contributions = static_cast<std::size_t>(particle_count) * 81u;
        g2p_pic_flip_particle_vjp_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, positions, particle_velocity, old_grid_velocity, new_grid_velocity, blend, velocity_adjoint, position_adjoint, particle_velocity_adjoint);
        check_launch("PIC FLIP particle VJP");
        g2p_pic_flip_grid_vjp_kernel<<<blocks(total_faces(configuration)), block_size, 0, stream>>>(configuration, contributions, blend, sorted_keys, sorted_ids, positions, velocity_adjoint, old_grid_velocity_adjoint, new_grid_velocity_adjoint);
        check_launch("PIC FLIP grid VJP");
        g2p_pic_flip_blend_values_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, positions, particle_velocity, old_grid_velocity, new_grid_velocity, velocity_adjoint, reduction_values);
        check_launch("PIC FLIP blend contribution VJP");
        std::size_t bytes = reduction_scratch_bytes;
        check(cub::DeviceReduce::Sum(reduction_scratch, bytes, reduction_values, reduction_result, particle_count, stream), "reduce PIC FLIP blend adjoint");
        accumulate_scalar_kernel<<<1, 1, 0, stream>>>(reduction_result, blend_adjoint);
        check_launch("PIC FLIP blend VJP");
    }

    void g2p_apic_forward(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstStaggeredField grid_velocity, const VectorField velocity, const MatrixField affine) {
        g2p_apic_forward_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, positions, grid_velocity, velocity, affine);
        check_launch("APIC G2P forward");
    }

    void g2p_apic_jvp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstStaggeredField grid_velocity, const ConstVectorField position_tangent, const ConstStaggeredField grid_velocity_tangent, const VectorField velocity_tangent, const MatrixField affine_tangent) {
        g2p_apic_jvp_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, positions, grid_velocity, position_tangent, grid_velocity_tangent, velocity_tangent, affine_tangent);
        check_launch("APIC G2P JVP");
    }

    void g2p_apic_vjp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const ConstVectorField positions, const ConstStaggeredField grid_velocity, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const ConstVectorAdjointField velocity_adjoint, const ConstMatrixAdjointField affine_adjoint, const VectorAdjointField position_adjoint, const StaggeredAdjointField grid_velocity_adjoint) {
        const std::size_t contributions = static_cast<std::size_t>(particle_count) * 81u;
        g2p_apic_particle_vjp_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, positions, grid_velocity, velocity_adjoint, affine_adjoint, position_adjoint);
        check_launch("APIC particle VJP");
        g2p_apic_grid_vjp_kernel<<<blocks(total_faces(configuration)), block_size, 0, stream>>>(configuration, contributions, positions, sorted_keys, sorted_ids, velocity_adjoint, affine_adjoint, grid_velocity_adjoint);
        check_launch("APIC grid VJP");
    }

    void advect_forward(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const float particle_radius, const std::uint32_t* cell_types, const ConstStaggeredField solid_velocity, const ConstVectorField positions, const ConstVectorField velocities, std::uint32_t* collision_masks, const VectorField unconstrained_positions, const VectorField next_positions, const VectorField next_velocities) {
        advect_forward_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, particle_radius, cell_types, solid_velocity, positions, velocities, collision_masks, unconstrained_positions, next_positions, next_velocities);
        check_launch("particle advection forward");
    }

    void advect_jvp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const float, const std::uint32_t* collision_masks, const ConstVectorField position_tangent, const ConstVectorField velocity_tangent, const VectorField next_position_tangent, const VectorField next_velocity_tangent) {
        advect_jvp_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, collision_masks, position_tangent, velocity_tangent, next_position_tangent, next_velocity_tangent);
        check_launch("particle advection JVP");
    }

    void advect_vjp(const cudaStream_t stream, const Configuration configuration, const std::uint32_t particle_count, const float, const std::uint32_t* collision_masks, const ConstVectorAdjointField next_position_adjoint, const ConstVectorAdjointField next_velocity_adjoint, const VectorAdjointField position_adjoint, const VectorAdjointField velocity_adjoint) {
        advect_vjp_kernel<<<blocks(particle_count), block_size, 0, stream>>>(configuration, particle_count, collision_masks, next_position_adjoint, next_velocity_adjoint, position_adjoint, velocity_adjoint);
        check_launch("particle advection VJP");
    }

    void accumulate_vector_adjoint(const cudaStream_t stream, const std::size_t count, const ConstVectorAdjointField source, const VectorAdjointField destination) {
        accumulate_double_kernel<<<blocks(count), block_size, 0, stream>>>(count, source.x, destination.x);
        accumulate_double_kernel<<<blocks(count), block_size, 0, stream>>>(count, source.y, destination.y);
        accumulate_double_kernel<<<blocks(count), block_size, 0, stream>>>(count, source.z, destination.z);
        check_launch("accumulate vector adjoint");
    }

    void accumulate_staggered_adjoint(const cudaStream_t stream, const std::size_t x_count, const std::size_t y_count, const std::size_t z_count, const ConstStaggeredAdjointField source, const StaggeredAdjointField destination) {
        accumulate_double_kernel<<<blocks(x_count), block_size, 0, stream>>>(x_count, source.x, destination.x);
        accumulate_double_kernel<<<blocks(y_count), block_size, 0, stream>>>(y_count, source.y, destination.y);
        accumulate_double_kernel<<<blocks(z_count), block_size, 0, stream>>>(z_count, source.z, destination.z);
        check_launch("accumulate staggered adjoint");
    }

    void accumulate_matrix_adjoint(const cudaStream_t stream, const std::size_t count, const double* const source[9], double* const destination[9]) {
        for (int component_index = 0; component_index < 9; ++component_index) accumulate_double_kernel<<<blocks(count), block_size, 0, stream>>>(count, source[component_index], destination[component_index]);
        check_launch("accumulate matrix adjoint");
    }

} // namespace xayah::fluid::grid::cuda_kernels
