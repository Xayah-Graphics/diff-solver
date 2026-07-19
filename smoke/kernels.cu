#include "kernels.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

namespace xayah::smoke::cuda_kernels {

    namespace {

        constexpr unsigned block_size = 256u;
        constexpr float smooth_epsilon = 1.0e-6F;

        struct Vector {
            float x;
            float y;
            float z;
        };

        struct Sample {
            float value;
            Vector gradient;
        };

        struct Trace {
            Vector position;
            Vector derivative;
        };

        __host__ unsigned blocks(const std::uint64_t count) {
            return static_cast<unsigned>((count + block_size - 1u) / block_size);
        }

        __host__ void check_launch(const char* name) {
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error(std::string{name} + ": " + cudaGetErrorString(status));
        }

        __host__ __device__ std::uint64_t cell_count(const Grid grid) {
            return static_cast<std::uint64_t>(grid.nx) * grid.ny * grid.nz;
        }

        __host__ __device__ int extent(const Grid grid, const int component_axis, const int dimension) {
            const int base = dimension == 0 ? grid.nx : dimension == 1 ? grid.ny : grid.nz;
            return base + (component_axis == dimension ? 1 : 0);
        }

        __host__ __device__ std::uint64_t face_count(const Grid grid, const int axis) {
            return static_cast<std::uint64_t>(extent(grid, axis, 0)) * extent(grid, axis, 1) * extent(grid, axis, 2);
        }

        __device__ std::uint64_t index3(const int x, const int y, const int z, const int nx, const int ny) {
            return static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(nx) * (static_cast<std::uint64_t>(y) + static_cast<std::uint64_t>(ny) * z);
        }

        __device__ void decode(const std::uint64_t index, const int nx, const int ny, int& x, int& y, int& z) {
            x = static_cast<int>(index % nx);
            const std::uint64_t yz = index / nx;
            y = static_cast<int>(yz % ny);
            z = static_cast<int>(yz / ny);
        }

        __host__ __device__ float* component(const StaggeredVectorView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const float* component(const ConstStaggeredVectorView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const float* component(const ConstCenteredVectorView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ double* component(const StaggeredVectorAdjointView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ const double* component(const ConstStaggeredVectorAdjointView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __host__ __device__ double* component(const CenteredVectorAdjointView field, const int axis) {
            return axis == 0 ? field.x : axis == 1 ? field.y : field.z;
        }

        __device__ bool periodic(const VelocityBoundaryData boundary, const int dimension) {
            return boundary.modes[dimension * 2] == 3u && boundary.modes[dimension * 2 + 1] == 3u;
        }

        __device__ bool periodic(const ScalarBoundaryData boundary, const int dimension) {
            return boundary.modes[dimension * 2] == 2u && boundary.modes[dimension * 2 + 1] == 2u;
        }

        __device__ int wrap(const int value, const int period) {
            const int remainder = value % period;
            return remainder < 0 ? remainder + period : remainder;
        }

        __device__ int map_coordinate(const int value, const int size, const bool is_periodic, const int period) {
            if (is_periodic) return wrap(value, period);
            return max(0, min(size - 1, value));
        }

        __device__ std::uint64_t mapped_cell_index(int x, int y, int z, const Grid grid, const bool periodic_x, const bool periodic_y, const bool periodic_z) {
            x = map_coordinate(x, grid.nx, periodic_x, grid.nx);
            y = map_coordinate(y, grid.ny, periodic_y, grid.ny);
            z = map_coordinate(z, grid.nz, periodic_z, grid.nz);
            return index3(x, y, z, grid.nx, grid.ny);
        }

        __device__ std::uint64_t mapped_face_index(int x, int y, int z, const Grid grid, const int axis, const VelocityBoundaryData boundary) {
            const int ex = extent(grid, axis, 0);
            const int ey = extent(grid, axis, 1);
            const int ez = extent(grid, axis, 2);
            x = map_coordinate(x, ex, periodic(boundary, 0), grid.nx);
            y = map_coordinate(y, ey, periodic(boundary, 1), grid.ny);
            z = map_coordinate(z, ez, periodic(boundary, 2), grid.nz);
            return index3(x, y, z, ex, ey);
        }

        __device__ float load_scalar(const float* values, const int x, const int y, const int z, const Grid grid, const ScalarBoundaryData boundary) {
            if (x < 0 && boundary.modes[0] == 0u) return boundary.values[0];
            if (x >= grid.nx && boundary.modes[1] == 0u) return boundary.values[1];
            if (y < 0 && boundary.modes[2] == 0u) return boundary.values[2];
            if (y >= grid.ny && boundary.modes[3] == 0u) return boundary.values[3];
            if (z < 0 && boundary.modes[4] == 0u) return boundary.values[4];
            if (z >= grid.nz && boundary.modes[5] == 0u) return boundary.values[5];
            return values[mapped_cell_index(x, y, z, grid, periodic(boundary, 0), periodic(boundary, 1), periodic(boundary, 2))];
        }

        __device__ float load_face(const float* values, const int axis, const int x, const int y, const int z, const Grid grid, const VelocityBoundaryData boundary) {
            const int coordinates[3]{x, y, z};
            for (int dimension = 0; dimension < 3; ++dimension) {
                const int size = extent(grid, axis, dimension);
                if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
                const int face = 2 * dimension + (coordinates[dimension] >= size);
                if (boundary.modes[face] == 0u || (boundary.modes[face] == 2u && axis == dimension)) return boundary.values[3 * face + axis];
            }
            return values[mapped_face_index(x, y, z, grid, axis, boundary)];
        }

        __device__ Sample sample_scalar(const float* values, const Vector position, const Grid grid, const ScalarBoundaryData boundary) {
            const float gx = position.x / grid.cell_size - 0.5F;
            const float gy = position.y / grid.cell_size - 0.5F;
            const float gz = position.z / grid.cell_size - 0.5F;
            const int x0 = static_cast<int>(floorf(gx));
            const int y0 = static_cast<int>(floorf(gy));
            const int z0 = static_cast<int>(floorf(gz));
            const float tx = gx - x0;
            const float ty = gy - y0;
            const float tz = gz - z0;
            float values8[2][2][2];
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) values8[dz][dy][dx] = load_scalar(values, x0 + dx, y0 + dy, z0 + dz, grid, boundary);
            const float wx[2]{1.0F - tx, tx};
            const float wy[2]{1.0F - ty, ty};
            const float wz[2]{1.0F - tz, tz};
            Sample result{};
            for (int dz = 0; dz < 2; ++dz) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) result.value += wx[dx] * wy[dy] * wz[dz] * values8[dz][dy][dx];
                    result.gradient.x += wy[dy] * wz[dz] * (values8[dz][dy][1] - values8[dz][dy][0]) / grid.cell_size;
                }
                for (int dx = 0; dx < 2; ++dx) result.gradient.y += wx[dx] * wz[dz] * (values8[dz][1][dx] - values8[dz][0][dx]) / grid.cell_size;
            }
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) result.gradient.z += wx[dx] * wy[dy] * (values8[1][dy][dx] - values8[0][dy][dx]) / grid.cell_size;
            return result;
        }

        __device__ Sample sample_face(const float* values, const int axis, const Vector position, const Grid grid, const VelocityBoundaryData boundary) {
            const float gx = position.x / grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
            const float gy = position.y / grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
            const float gz = position.z / grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
            const int x0 = static_cast<int>(floorf(gx));
            const int y0 = static_cast<int>(floorf(gy));
            const int z0 = static_cast<int>(floorf(gz));
            const float tx = gx - x0;
            const float ty = gy - y0;
            const float tz = gz - z0;
            float values8[2][2][2];
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) values8[dz][dy][dx] = load_face(values, axis, x0 + dx, y0 + dy, z0 + dz, grid, boundary);
            const float wx[2]{1.0F - tx, tx};
            const float wy[2]{1.0F - ty, ty};
            const float wz[2]{1.0F - tz, tz};
            Sample result{};
            for (int dz = 0; dz < 2; ++dz) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) result.value += wx[dx] * wy[dy] * wz[dz] * values8[dz][dy][dx];
                    result.gradient.x += wy[dy] * wz[dz] * (values8[dz][dy][1] - values8[dz][dy][0]) / grid.cell_size;
                }
                for (int dx = 0; dx < 2; ++dx) result.gradient.y += wx[dx] * wz[dz] * (values8[dz][1][dx] - values8[dz][0][dx]) / grid.cell_size;
            }
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) result.gradient.z += wx[dx] * wy[dy] * (values8[1][dy][dx] - values8[0][dy][dx]) / grid.cell_size;
            return result;
        }

        __device__ void scatter_scalar(double* values, const Vector position, const double adjoint, const Grid grid, const ScalarBoundaryData boundary) {
            const float gx = position.x / grid.cell_size - 0.5F;
            const float gy = position.y / grid.cell_size - 0.5F;
            const float gz = position.z / grid.cell_size - 0.5F;
            const int x0 = static_cast<int>(floorf(gx));
            const int y0 = static_cast<int>(floorf(gy));
            const int z0 = static_cast<int>(floorf(gz));
            const double wx[2]{1.0 - (gx - x0), gx - x0};
            const double wy[2]{1.0 - (gy - y0), gy - y0};
            const double wz[2]{1.0 - (gz - z0), gz - z0};
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int x = x0 + dx;
                        const int y = y0 + dy;
                        const int z = z0 + dz;
                        if ((x < 0 && boundary.modes[0] == 0u) || (x >= grid.nx && boundary.modes[1] == 0u) || (y < 0 && boundary.modes[2] == 0u) || (y >= grid.ny && boundary.modes[3] == 0u) || (z < 0 && boundary.modes[4] == 0u) || (z >= grid.nz && boundary.modes[5] == 0u)) continue;
                        atomicAdd(values + mapped_cell_index(x, y, z, grid, periodic(boundary, 0), periodic(boundary, 1), periodic(boundary, 2)), adjoint * wx[dx] * wy[dy] * wz[dz]);
                    }
        }

        __device__ void scatter_face(double* values, const int axis, const Vector position, const double adjoint, const Grid grid, const VelocityBoundaryData boundary) {
            const float gx = position.x / grid.cell_size - (axis == 0 ? 0.0F : 0.5F);
            const float gy = position.y / grid.cell_size - (axis == 1 ? 0.0F : 0.5F);
            const float gz = position.z / grid.cell_size - (axis == 2 ? 0.0F : 0.5F);
            const int x0 = static_cast<int>(floorf(gx));
            const int y0 = static_cast<int>(floorf(gy));
            const int z0 = static_cast<int>(floorf(gz));
            const double wx[2]{1.0 - (gx - x0), gx - x0};
            const double wy[2]{1.0 - (gy - y0), gy - y0};
            const double wz[2]{1.0 - (gz - z0), gz - z0};
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int coordinates[3]{x0 + dx, y0 + dy, z0 + dz};
                        bool constant = false;
                        for (int dimension = 0; dimension < 3; ++dimension) {
                            const int size = extent(grid, axis, dimension);
                            if (coordinates[dimension] >= 0 && coordinates[dimension] < size) continue;
                            const int face = 2 * dimension + (coordinates[dimension] >= size);
                            if (boundary.modes[face] == 0u || (boundary.modes[face] == 2u && axis == dimension)) constant = true;
                        }
                        if (!constant) atomicAdd(values + mapped_face_index(coordinates[0], coordinates[1], coordinates[2], grid, axis, boundary), adjoint * wx[dx] * wy[dy] * wz[dz]);
                    }
        }

        __device__ Vector sample_velocity_value(const ConstStaggeredVectorView velocity, const Vector position, const Grid grid, const VelocityBoundaryData boundary) {
            return {sample_face(velocity.x, 0, position, grid, boundary).value, sample_face(velocity.y, 1, position, grid, boundary).value, sample_face(velocity.z, 2, position, grid, boundary).value};
        }

        __device__ bool solid_at(Vector position, const std::uint32_t* cell_mask, const Grid grid, const VelocityBoundaryData boundary) {
            if (periodic(boundary, 0)) position.x -= floorf(position.x / (grid.nx * grid.cell_size)) * grid.nx * grid.cell_size;
            if (periodic(boundary, 1)) position.y -= floorf(position.y / (grid.ny * grid.cell_size)) * grid.ny * grid.cell_size;
            if (periodic(boundary, 2)) position.z -= floorf(position.z / (grid.nz * grid.cell_size)) * grid.nz * grid.cell_size;
            const int x = max(0, min(grid.nx - 1, static_cast<int>(floorf(position.x / grid.cell_size))));
            const int y = max(0, min(grid.ny - 1, static_cast<int>(floorf(position.y / grid.cell_size))));
            const int z = max(0, min(grid.nz - 1, static_cast<int>(floorf(position.z / grid.cell_size))));
            return cell_mask[index3(x, y, z, grid.nx, grid.ny)] != 0u;
        }

        __device__ Trace trace_rk2(const Vector start, const ConstStaggeredVectorView velocity, const std::uint32_t* cell_mask, const Grid grid, const VelocityBoundaryData boundary) {
            const Vector value0 = sample_velocity_value(velocity, start, grid, boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Vector value1 = sample_velocity_value(velocity, midpoint, grid, boundary);
            Vector raw{start.x - grid.time_step * value1.x, start.y - grid.time_step * value1.y, start.z - grid.time_step * value1.z};
            Vector derivative{1.0F, 1.0F, 1.0F};
            const Vector maximum{grid.nx * grid.cell_size, grid.ny * grid.cell_size, grid.nz * grid.cell_size};
            if (!periodic(boundary, 0) && (raw.x < 0.0F || raw.x > maximum.x)) { raw.x = fminf(fmaxf(raw.x, 0.0F), maximum.x); derivative.x = 0.0F; }
            if (!periodic(boundary, 1) && (raw.y < 0.0F || raw.y > maximum.y)) { raw.y = fminf(fmaxf(raw.y, 0.0F), maximum.y); derivative.y = 0.0F; }
            if (!periodic(boundary, 2) && (raw.z < 0.0F || raw.z > maximum.z)) { raw.z = fminf(fmaxf(raw.z, 0.0F), maximum.z); derivative.z = 0.0F; }
            if (!solid_at(raw, cell_mask, grid, boundary)) return {raw, derivative};
            float lo = 0.0F;
            float hi = 1.0F;
            for (int iteration = 0; iteration < 8; ++iteration) {
                const float fraction = 0.5F * (lo + hi);
                const Vector test{start.x + fraction * (raw.x - start.x), start.y + fraction * (raw.y - start.y), start.z + fraction * (raw.z - start.z)};
                if (solid_at(test, cell_mask, grid, boundary)) hi = fraction;
                else lo = fraction;
            }
            return {{start.x + lo * (raw.x - start.x), start.y + lo * (raw.y - start.y), start.z + lo * (raw.z - start.z)}, {lo * derivative.x, lo * derivative.y, lo * derivative.z}};
        }

        __device__ Vector face_position(const int axis, const int x, const int y, const int z, const Grid grid) {
            return {(x + (axis == 0 ? 0.0F : 0.5F)) * grid.cell_size, (y + (axis == 1 ? 0.0F : 0.5F)) * grid.cell_size, (z + (axis == 2 ? 0.0F : 0.5F)) * grid.cell_size};
        }

        __device__ Vector cell_position(const int x, const int y, const int z, const Grid grid) {
            return {(x + 0.5F) * grid.cell_size, (y + 0.5F) * grid.cell_size, (z + 0.5F) * grid.cell_size};
        }

        __global__ void source_forward_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* state, const float* source, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            output[index] = cell_mask[index] == 0u ? state[index] + grid.time_step * source[index] : 0.0F;
        }

        __global__ void accumulate_kernel(const double* source, double* destination, const std::uint64_t count) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count) destination[index] += source[index];
        }

        __global__ void source_vjp_kernel(const Grid grid, const std::uint32_t* cell_mask, const double* output_adjoint, double* state_adjoint, double* source_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            state_adjoint[index] += output_adjoint[index];
            source_adjoint[index] += grid.time_step * output_adjoint[index];
        }

        __global__ void buoyancy_forward_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* density, const float* temperature, const ConstCenteredVectorView external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const CenteredVectorView force) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (cell_mask[index] != 0u) {
                force.x[index] = 0.0F; force.y[index] = 0.0F; force.z[index] = 0.0F; return;
            }
            force.x[index] = external_acceleration.x[index];
            force.y[index] = external_acceleration.y[index] + density_buoyancy[0] * density[index] + temperature_buoyancy[0] * (temperature[index] - ambient_temperature[0]);
            force.z[index] = external_acceleration.z[index];
        }

        __global__ void buoyancy_jvp_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* density, const float* temperature, const float* density_tangent, const float* temperature_tangent, const ConstCenteredVectorView external_tangent, const float* ambient, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, const CenteredVectorView output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (cell_mask[index] != 0u) {
                output.x[index] = 0.0F; output.y[index] = 0.0F; output.z[index] = 0.0F; return;
            }
            output.x[index] = external_tangent.x[index];
            output.y[index] = external_tangent.y[index] + density_buoyancy[0] * density_tangent[index] + density_buoyancy_tangent[0] * density[index] + temperature_buoyancy[0] * (temperature_tangent[index] - ambient_tangent[0]) + temperature_buoyancy_tangent[0] * (temperature[index] - ambient[0]);
            output.z[index] = external_tangent.z[index];
        }

        __global__ void buoyancy_vjp_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* density, const float* temperature, const float* ambient, const float* density_buoyancy, const float* temperature_buoyancy, const ConstCenteredVectorAdjointView force_adjoint, double* density_adjoint, double* temperature_adjoint, const CenteredVectorAdjointView external_adjoint, double* ambient_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            external_adjoint.x[index] += force_adjoint.x[index];
            external_adjoint.y[index] += force_adjoint.y[index];
            external_adjoint.z[index] += force_adjoint.z[index];
            density_adjoint[index] += density_buoyancy[0] * force_adjoint.y[index];
            temperature_adjoint[index] += temperature_buoyancy[0] * force_adjoint.y[index];
            atomicAdd(ambient_adjoint, -temperature_buoyancy[0] * force_adjoint.y[index]);
            atomicAdd(density_buoyancy_adjoint, density[index] * force_adjoint.y[index]);
            atomicAdd(temperature_buoyancy_adjoint, (temperature[index] - ambient[0]) * force_adjoint.y[index]);
        }

        __device__ float averaged_center_force(const float* force, const int axis, const int x, const int y, const int z, const Grid grid, const std::uint32_t* cell_mask) {
            int first_x = x, first_y = y, first_z = z;
            int second_x = x, second_y = y, second_z = z;
            if (axis == 0) --first_x;
            if (axis == 1) --first_y;
            if (axis == 2) --first_z;
            float sum = 0.0F;
            float count = 0.0F;
            if (first_x >= 0 && first_x < grid.nx && first_y >= 0 && first_y < grid.ny && first_z >= 0 && first_z < grid.nz) {
                const std::uint64_t cell = index3(first_x, first_y, first_z, grid.nx, grid.ny);
                if (cell_mask[cell] == 0u) { sum += force[cell]; count += 1.0F; }
            }
            if (second_x >= 0 && second_x < grid.nx && second_y >= 0 && second_y < grid.ny && second_z >= 0 && second_z < grid.nz) {
                const std::uint64_t cell = index3(second_x, second_y, second_z, grid.nx, grid.ny);
                if (cell_mask[cell] == 0u) { sum += force[cell]; count += 1.0F; }
            }
            return count == 0.0F ? 0.0F : sum / count;
        }

        __global__ void integrate_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const float* velocity, const float* force, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            output[index] = velocity[index] + grid.time_step * averaged_center_force(force, axis, x, y, z, grid, cell_mask);
        }

        __global__ void integrate_vjp_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const double* output_adjoint, double* velocity_adjoint, double* force_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            velocity_adjoint[index] += output_adjoint[index];
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            int cells[2][3]{{x - (axis == 0), y - (axis == 1), z - (axis == 2)}, {x, y, z}};
            int count = 0;
            for (int side = 0; side < 2; ++side)
                if (cells[side][0] >= 0 && cells[side][0] < grid.nx && cells[side][1] >= 0 && cells[side][1] < grid.ny && cells[side][2] >= 0 && cells[side][2] < grid.nz && cell_mask[index3(cells[side][0], cells[side][1], cells[side][2], grid.nx, grid.ny)] == 0u) ++count;
            if (count == 0) return;
            for (int side = 0; side < 2; ++side)
                if (cells[side][0] >= 0 && cells[side][0] < grid.nx && cells[side][1] >= 0 && cells[side][1] < grid.ny && cells[side][2] >= 0 && cells[side][2] < grid.nz && cell_mask[index3(cells[side][0], cells[side][1], cells[side][2], grid.nx, grid.ny)] == 0u) atomicAdd(force_adjoint + index3(cells[side][0], cells[side][1], cells[side][2], grid.nx, grid.ny), grid.time_step * output_adjoint[index] / count);
        }

        __global__ void advect_velocity_forward_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            const Trace trace = trace_rk2(face_position(axis, x, y, z, grid), velocity, cell_mask, grid, boundary);
            output[index] = sample_face(component(velocity, axis), axis, trace.position, grid, boundary).value;
        }

        __global__ void advect_velocity_jvp_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const ConstStaggeredVectorView velocity_tangent, const VelocityBoundaryData boundary, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            const Vector start = face_position(axis, x, y, z, grid);
            VelocityBoundaryData tangent_boundary = boundary;
            for (float& value : tangent_boundary.values) value = 0.0F;
            const Vector value0 = sample_velocity_value(velocity, start, grid, boundary);
            const Vector tangent0 = sample_velocity_value(velocity_tangent, start, grid, tangent_boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Vector midpoint_tangent{-0.5F * grid.time_step * tangent0.x, -0.5F * grid.time_step * tangent0.y, -0.5F * grid.time_step * tangent0.z};
            Sample velocity_samples[3]{sample_face(velocity.x, 0, midpoint, grid, boundary), sample_face(velocity.y, 1, midpoint, grid, boundary), sample_face(velocity.z, 2, midpoint, grid, boundary)};
            const Vector sampled_tangent = sample_velocity_value(velocity_tangent, midpoint, grid, tangent_boundary);
            Vector value1_tangent{
                sampled_tangent.x + velocity_samples[0].gradient.x * midpoint_tangent.x + velocity_samples[0].gradient.y * midpoint_tangent.y + velocity_samples[0].gradient.z * midpoint_tangent.z,
                sampled_tangent.y + velocity_samples[1].gradient.x * midpoint_tangent.x + velocity_samples[1].gradient.y * midpoint_tangent.y + velocity_samples[1].gradient.z * midpoint_tangent.z,
                sampled_tangent.z + velocity_samples[2].gradient.x * midpoint_tangent.x + velocity_samples[2].gradient.y * midpoint_tangent.y + velocity_samples[2].gradient.z * midpoint_tangent.z,
            };
            const Trace trace = trace_rk2(start, velocity, cell_mask, grid, boundary);
            const Vector position_tangent{-trace.derivative.x * grid.time_step * value1_tangent.x, -trace.derivative.y * grid.time_step * value1_tangent.y, -trace.derivative.z * grid.time_step * value1_tangent.z};
            const Sample source_sample = sample_face(component(velocity, axis), axis, trace.position, grid, boundary);
            output_tangent[index] = sample_face(component(velocity_tangent, axis), axis, trace.position, grid, tangent_boundary).value + source_sample.gradient.x * position_tangent.x + source_sample.gradient.y * position_tangent.y + source_sample.gradient.z * position_tangent.z;
        }

        __global__ void advect_velocity_vjp_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, const double* output_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            const Vector start = face_position(axis, x, y, z, grid);
            const Vector value0 = sample_velocity_value(velocity, start, grid, boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Trace trace = trace_rk2(start, velocity, cell_mask, grid, boundary);
            const Sample source_sample = sample_face(component(velocity, axis), axis, trace.position, grid, boundary);
            scatter_face(component(velocity_adjoint, axis), axis, trace.position, output_adjoint[index], grid, boundary);
            const double value1_adjoint_x = -static_cast<double>(trace.derivative.x * grid.time_step * source_sample.gradient.x) * output_adjoint[index];
            const double value1_adjoint_y = -static_cast<double>(trace.derivative.y * grid.time_step * source_sample.gradient.y) * output_adjoint[index];
            const double value1_adjoint_z = -static_cast<double>(trace.derivative.z * grid.time_step * source_sample.gradient.z) * output_adjoint[index];
            Sample midpoint_samples[3]{sample_face(velocity.x, 0, midpoint, grid, boundary), sample_face(velocity.y, 1, midpoint, grid, boundary), sample_face(velocity.z, 2, midpoint, grid, boundary)};
            scatter_face(velocity_adjoint.x, 0, midpoint, value1_adjoint_x, grid, boundary);
            scatter_face(velocity_adjoint.y, 1, midpoint, value1_adjoint_y, grid, boundary);
            scatter_face(velocity_adjoint.z, 2, midpoint, value1_adjoint_z, grid, boundary);
            const double midpoint_adjoint_x = midpoint_samples[0].gradient.x * value1_adjoint_x + midpoint_samples[1].gradient.x * value1_adjoint_y + midpoint_samples[2].gradient.x * value1_adjoint_z;
            const double midpoint_adjoint_y = midpoint_samples[0].gradient.y * value1_adjoint_x + midpoint_samples[1].gradient.y * value1_adjoint_y + midpoint_samples[2].gradient.y * value1_adjoint_z;
            const double midpoint_adjoint_z = midpoint_samples[0].gradient.z * value1_adjoint_x + midpoint_samples[1].gradient.z * value1_adjoint_y + midpoint_samples[2].gradient.z * value1_adjoint_z;
            scatter_face(velocity_adjoint.x, 0, start, -0.5 * grid.time_step * midpoint_adjoint_x, grid, boundary);
            scatter_face(velocity_adjoint.y, 1, start, -0.5 * grid.time_step * midpoint_adjoint_y, grid, boundary);
            scatter_face(velocity_adjoint.z, 2, start, -0.5 * grid.time_step * midpoint_adjoint_z, grid, boundary);
        }

        __device__ bool adjacent_solid(const int axis, const int x, const int y, const int z, const Grid grid, const std::uint32_t* cell_mask) {
            int first[3]{x - (axis == 0), y - (axis == 1), z - (axis == 2)};
            int second[3]{x, y, z};
            for (int side = 0; side < 2; ++side) {
                const int* cell = side == 0 ? first : second;
                if (cell[0] >= 0 && cell[0] < grid.nx && cell[1] >= 0 && cell[1] < grid.ny && cell[2] >= 0 && cell[2] < grid.nz && cell_mask[index3(cell[0], cell[1], cell[2], grid.nx, grid.ny)] != 0u) return true;
            }
            return false;
        }

        __device__ bool constrained_boundary(const int axis, const int x, const int y, const int z, const Grid grid, const VelocityBoundaryData boundary) {
            const int coordinate = axis == 0 ? x : axis == 1 ? y : z;
            const int maximum = axis == 0 ? grid.nx : axis == 1 ? grid.ny : grid.nz;
            if (coordinate != 0 && coordinate != maximum) return false;
            const std::uint32_t mode = boundary.modes[axis * 2 + (coordinate == maximum)];
            return mode == 0u || mode == 2u;
        }

        __device__ float boundary_velocity_value(const int axis, const int x, const int y, const int z, const Grid grid, const VelocityBoundaryData boundary) {
            const int coordinate = axis == 0 ? x : axis == 1 ? y : z;
            const int maximum = axis == 0 ? grid.nx : axis == 1 ? grid.ny : grid.nz;
            const int face = axis * 2 + (coordinate == maximum);
            return boundary.values[face * 3 + axis];
        }

        __global__ void constrain_velocity_forward_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const float* collider_velocity, const float* velocity, const VelocityBoundaryData boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            if (constrained_boundary(axis, x, y, z, grid, boundary)) { output[index] = boundary_velocity_value(axis, x, y, z, grid, boundary); return; }
            if (adjacent_solid(axis, x, y, z, grid, cell_mask)) { output[index] = collider_velocity == nullptr ? 0.0F : collider_velocity[index]; return; }
            if (periodic(boundary, axis) && (axis == 0 ? x == grid.nx : axis == 1 ? y == grid.ny : z == grid.nz)) {
                if (axis == 0) x = 0;
                if (axis == 1) y = 0;
                if (axis == 2) z = 0;
                output[index] = velocity[index3(x, y, z, extent(grid, axis, 0), extent(grid, axis, 1))];
                return;
            }
            output[index] = velocity[index];
        }

        __global__ void constrain_velocity_vjp_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const double* output_adjoint, const VelocityBoundaryData boundary, double* velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            if (constrained_boundary(axis, x, y, z, grid, boundary) || adjacent_solid(axis, x, y, z, grid, cell_mask)) return;
            if (periodic(boundary, axis) && (axis == 0 ? x == grid.nx : axis == 1 ? y == grid.ny : z == grid.nz)) {
                if (axis == 0) x = 0;
                if (axis == 1) y = 0;
                if (axis == 2) z = 0;
                atomicAdd(velocity_adjoint + index3(x, y, z, extent(grid, axis, 0), extent(grid, axis, 1)), output_adjoint[index]);
                return;
            }
            atomicAdd(velocity_adjoint + index, output_adjoint[index]);
        }

        __global__ void advect_scalar_forward_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* collider_value, const float* source, const ConstStaggeredVectorView velocity, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (cell_mask[index] != 0u) { output[index] = collider_value[index]; return; }
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const Trace trace = trace_rk2(cell_position(x, y, z, grid), velocity, cell_mask, grid, velocity_boundary);
            output[index] = sample_scalar(source, trace.position, grid, scalar_boundary).value;
        }

        __global__ void advect_scalar_jvp_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* source, const float* source_tangent, const ConstStaggeredVectorView velocity, const ConstStaggeredVectorView velocity_tangent, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, float* output_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (cell_mask[index] != 0u) { output_tangent[index] = 0.0F; return; }
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const Vector start = cell_position(x, y, z, grid);
            ScalarBoundaryData scalar_tangent_boundary = scalar_boundary;
            for (float& value : scalar_tangent_boundary.values) value = 0.0F;
            VelocityBoundaryData velocity_tangent_boundary = velocity_boundary;
            for (float& value : velocity_tangent_boundary.values) value = 0.0F;
            const Vector value0 = sample_velocity_value(velocity, start, grid, velocity_boundary);
            const Vector tangent0 = sample_velocity_value(velocity_tangent, start, grid, velocity_tangent_boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Vector midpoint_tangent{-0.5F * grid.time_step * tangent0.x, -0.5F * grid.time_step * tangent0.y, -0.5F * grid.time_step * tangent0.z};
            Sample velocity_samples[3]{sample_face(velocity.x, 0, midpoint, grid, velocity_boundary), sample_face(velocity.y, 1, midpoint, grid, velocity_boundary), sample_face(velocity.z, 2, midpoint, grid, velocity_boundary)};
            const Vector sampled_tangent = sample_velocity_value(velocity_tangent, midpoint, grid, velocity_tangent_boundary);
            const Vector value1_tangent{
                sampled_tangent.x + velocity_samples[0].gradient.x * midpoint_tangent.x + velocity_samples[0].gradient.y * midpoint_tangent.y + velocity_samples[0].gradient.z * midpoint_tangent.z,
                sampled_tangent.y + velocity_samples[1].gradient.x * midpoint_tangent.x + velocity_samples[1].gradient.y * midpoint_tangent.y + velocity_samples[1].gradient.z * midpoint_tangent.z,
                sampled_tangent.z + velocity_samples[2].gradient.x * midpoint_tangent.x + velocity_samples[2].gradient.y * midpoint_tangent.y + velocity_samples[2].gradient.z * midpoint_tangent.z,
            };
            const Trace trace = trace_rk2(start, velocity, cell_mask, grid, velocity_boundary);
            const Vector position_tangent{-trace.derivative.x * grid.time_step * value1_tangent.x, -trace.derivative.y * grid.time_step * value1_tangent.y, -trace.derivative.z * grid.time_step * value1_tangent.z};
            const Sample source_sample = sample_scalar(source, trace.position, grid, scalar_boundary);
            output_tangent[index] = sample_scalar(source_tangent, trace.position, grid, scalar_tangent_boundary).value + source_sample.gradient.x * position_tangent.x + source_sample.gradient.y * position_tangent.y + source_sample.gradient.z * position_tangent.z;
        }

        __global__ void advect_scalar_vjp_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* source, const ConstStaggeredVectorView velocity, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, const double* output_adjoint, double* source_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const Vector start = cell_position(x, y, z, grid);
            const Vector value0 = sample_velocity_value(velocity, start, grid, velocity_boundary);
            const Vector midpoint{start.x - 0.5F * grid.time_step * value0.x, start.y - 0.5F * grid.time_step * value0.y, start.z - 0.5F * grid.time_step * value0.z};
            const Trace trace = trace_rk2(start, velocity, cell_mask, grid, velocity_boundary);
            const Sample source_sample = sample_scalar(source, trace.position, grid, scalar_boundary);
            scatter_scalar(source_adjoint, trace.position, output_adjoint[index], grid, scalar_boundary);
            const double value1_adjoint_x = -static_cast<double>(trace.derivative.x * grid.time_step * source_sample.gradient.x) * output_adjoint[index];
            const double value1_adjoint_y = -static_cast<double>(trace.derivative.y * grid.time_step * source_sample.gradient.y) * output_adjoint[index];
            const double value1_adjoint_z = -static_cast<double>(trace.derivative.z * grid.time_step * source_sample.gradient.z) * output_adjoint[index];
            Sample midpoint_samples[3]{sample_face(velocity.x, 0, midpoint, grid, velocity_boundary), sample_face(velocity.y, 1, midpoint, grid, velocity_boundary), sample_face(velocity.z, 2, midpoint, grid, velocity_boundary)};
            scatter_face(velocity_adjoint.x, 0, midpoint, value1_adjoint_x, grid, velocity_boundary);
            scatter_face(velocity_adjoint.y, 1, midpoint, value1_adjoint_y, grid, velocity_boundary);
            scatter_face(velocity_adjoint.z, 2, midpoint, value1_adjoint_z, grid, velocity_boundary);
            const double midpoint_adjoint_x = midpoint_samples[0].gradient.x * value1_adjoint_x + midpoint_samples[1].gradient.x * value1_adjoint_y + midpoint_samples[2].gradient.x * value1_adjoint_z;
            const double midpoint_adjoint_y = midpoint_samples[0].gradient.y * value1_adjoint_x + midpoint_samples[1].gradient.y * value1_adjoint_y + midpoint_samples[2].gradient.y * value1_adjoint_z;
            const double midpoint_adjoint_z = midpoint_samples[0].gradient.z * value1_adjoint_x + midpoint_samples[1].gradient.z * value1_adjoint_y + midpoint_samples[2].gradient.z * value1_adjoint_z;
            scatter_face(velocity_adjoint.x, 0, start, -0.5 * grid.time_step * midpoint_adjoint_x, grid, velocity_boundary);
            scatter_face(velocity_adjoint.y, 1, start, -0.5 * grid.time_step * midpoint_adjoint_y, grid, velocity_boundary);
            scatter_face(velocity_adjoint.z, 2, start, -0.5 * grid.time_step * midpoint_adjoint_z, grid, velocity_boundary);
        }

    } // namespace

    void accumulate(const cudaStream_t stream, const double* source, double* destination, const std::uint64_t count) {
        accumulate_kernel<<<blocks(count), block_size, 0, stream>>>(source, destination, count);
        check_launch("accumulate_kernel");
    }

    void source_forward(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView state, const ConstScalarView source, const ScalarView output) {
        source_forward_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, state.values, source.values, output.values);
        check_launch("source_forward_kernel");
    }

    void source_jvp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView state_tangent, const ConstScalarView source_tangent, const ScalarView output_tangent) {
        source_forward(stream, grid, cell_mask, state_tangent, source_tangent, output_tangent);
    }

    void source_vjp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarAdjointView output_adjoint, const ScalarAdjointView state_adjoint, const ScalarAdjointView source_adjoint) {
        source_vjp_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, output_adjoint.values, state_adjoint.values, source_adjoint.values);
        check_launch("source_vjp_kernel");
    }

    void buoyancy_forward(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView density, const ConstScalarView temperature, const ConstCenteredVectorView external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const CenteredVectorView force) {
        buoyancy_forward_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, density.values, temperature.values, external_acceleration, ambient_temperature, density_buoyancy, temperature_buoyancy, force);
        check_launch("buoyancy_forward_kernel");
    }

    void buoyancy_jvp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView density, const ConstScalarView temperature, const ConstScalarView density_tangent, const ConstScalarView temperature_tangent, const ConstCenteredVectorView external_acceleration_tangent, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_temperature_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, const CenteredVectorView force_tangent) {
        buoyancy_jvp_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, density.values, temperature.values, density_tangent.values, temperature_tangent.values, external_acceleration_tangent, ambient_temperature, density_buoyancy, temperature_buoyancy, ambient_temperature_tangent, density_buoyancy_tangent, temperature_buoyancy_tangent, force_tangent);
        check_launch("buoyancy_jvp_kernel");
    }

    void buoyancy_vjp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView density, const ConstScalarView temperature, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const ConstCenteredVectorAdjointView force_adjoint, const ScalarAdjointView density_adjoint, const ScalarAdjointView temperature_adjoint, const CenteredVectorAdjointView external_acceleration_adjoint, double* ambient_temperature_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint) {
        buoyancy_vjp_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, density.values, temperature.values, ambient_temperature, density_buoyancy, temperature_buoyancy, force_adjoint, density_adjoint.values, temperature_adjoint.values, external_acceleration_adjoint, ambient_temperature_adjoint, density_buoyancy_adjoint, temperature_buoyancy_adjoint);
        check_launch("buoyancy_vjp_kernel");
    }

    void integrate_velocity_forward(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const ConstCenteredVectorView force, const StaggeredVectorView output) {
        for (int axis = 0; axis < 3; ++axis) integrate_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, component(velocity, axis), component(force, axis), component(output, axis));
        check_launch("integrate_kernel");
    }

    void integrate_velocity_jvp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity_tangent, const ConstCenteredVectorView force_tangent, const StaggeredVectorView output_tangent) {
        integrate_velocity_forward(stream, grid, cell_mask, velocity_tangent, force_tangent, output_tangent);
    }

    void integrate_velocity_vjp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorAdjointView output_adjoint, const StaggeredVectorAdjointView velocity_adjoint, const CenteredVectorAdjointView force_adjoint) {
        for (int axis = 0; axis < 3; ++axis) integrate_vjp_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, component(output_adjoint, axis), component(velocity_adjoint, axis), component(force_adjoint, axis));
        check_launch("integrate_vjp_kernel");
    }

    void advect_velocity_forward(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, const StaggeredVectorView output) {
        for (int axis = 0; axis < 3; ++axis) advect_velocity_forward_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, velocity, boundary, component(output, axis));
        check_launch("advect_velocity_forward_kernel");
    }

    void advect_velocity_jvp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const ConstStaggeredVectorView velocity_tangent, const VelocityBoundaryData boundary, const StaggeredVectorView output_tangent) {
        for (int axis = 0; axis < 3; ++axis) advect_velocity_jvp_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, velocity, velocity_tangent, boundary, component(output_tangent, axis));
        check_launch("advect_velocity_jvp_kernel");
    }

    void advect_velocity_vjp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, const ConstStaggeredVectorAdjointView output_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
        for (int axis = 0; axis < 3; ++axis) advect_velocity_vjp_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, velocity, boundary, component(output_adjoint, axis), velocity_adjoint);
        check_launch("advect_velocity_vjp_kernel");
    }

    void constrain_velocity_forward(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView collider_velocity, const ConstStaggeredVectorView velocity, const VelocityBoundaryData boundary, const StaggeredVectorView output) {
        for (int axis = 0; axis < 3; ++axis) constrain_velocity_forward_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, component(collider_velocity, axis), component(velocity, axis), boundary, component(output, axis));
        check_launch("constrain_velocity_forward_kernel");
    }

    void constrain_velocity_jvp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity_tangent, const VelocityBoundaryData boundary, const StaggeredVectorView output_tangent) {
        ConstStaggeredVectorView zero{};
        VelocityBoundaryData tangent_boundary = boundary;
        for (float& value : tangent_boundary.values) value = 0.0F;
        constrain_velocity_forward(stream, grid, cell_mask, zero, velocity_tangent, tangent_boundary, output_tangent);
    }

    void constrain_velocity_vjp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorAdjointView output_adjoint, const VelocityBoundaryData boundary, const StaggeredVectorAdjointView velocity_adjoint) {
        for (int axis = 0; axis < 3; ++axis) constrain_velocity_vjp_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, component(output_adjoint, axis), boundary, component(velocity_adjoint, axis));
        check_launch("constrain_velocity_vjp_kernel");
    }

    void advect_scalar_forward(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView collider_value, const ConstScalarView source, const ConstStaggeredVectorView velocity, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, const ScalarView output) {
        advect_scalar_forward_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, collider_value.values, source.values, velocity, scalar_boundary, velocity_boundary, output.values);
        check_launch("advect_scalar_forward_kernel");
    }

    void advect_scalar_jvp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView source, const ConstScalarView source_tangent, const ConstStaggeredVectorView velocity, const ConstStaggeredVectorView velocity_tangent, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, const ScalarView output_tangent) {
        advect_scalar_jvp_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, source.values, source_tangent.values, velocity, velocity_tangent, scalar_boundary, velocity_boundary, output_tangent.values);
        check_launch("advect_scalar_jvp_kernel");
    }

    void advect_scalar_vjp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstScalarView source, const ConstStaggeredVectorView velocity, const ScalarBoundaryData scalar_boundary, const VelocityBoundaryData velocity_boundary, const ConstScalarAdjointView output_adjoint, const ScalarAdjointView source_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
        advect_scalar_vjp_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, source.values, velocity, scalar_boundary, velocity_boundary, output_adjoint.values, source_adjoint.values, velocity_adjoint);
        check_launch("advect_scalar_vjp_kernel");
    }

    namespace {

        __device__ float centered_load(const float* values, int x, int y, int z, const Grid grid) {
            x = max(0, min(grid.nx - 1, x));
            y = max(0, min(grid.ny - 1, y));
            z = max(0, min(grid.nz - 1, z));
            return values[index3(x, y, z, grid.nx, grid.ny)];
        }

        __device__ void centered_scatter(double* values, int x, int y, int z, const Grid grid, const double adjoint) {
            x = max(0, min(grid.nx - 1, x));
            y = max(0, min(grid.ny - 1, y));
            z = max(0, min(grid.nz - 1, z));
            atomicAdd(values + index3(x, y, z, grid.nx, grid.ny), adjoint);
        }

        __global__ void centered_velocity_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const CenteredVectorView centered) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (cell_mask[index] != 0u) { centered.x[index] = 0.0F; centered.y[index] = 0.0F; centered.z[index] = 0.0F; return; }
            centered.x[index] = 0.5F * (velocity.x[index3(x, y, z, grid.nx + 1, grid.ny)] + velocity.x[index3(x + 1, y, z, grid.nx + 1, grid.ny)]);
            centered.y[index] = 0.5F * (velocity.y[index3(x, y, z, grid.nx, grid.ny + 1)] + velocity.y[index3(x, y + 1, z, grid.nx, grid.ny + 1)]);
            centered.z[index] = 0.5F * (velocity.z[index3(x, y, z, grid.nx, grid.ny)] + velocity.z[index3(x, y, z + 1, grid.nx, grid.ny)]);
        }

        __global__ void curl_magnitude_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView centered, const CenteredVectorView vorticity, float* magnitude) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (cell_mask[index] != 0u) { vorticity.x[index] = 0.0F; vorticity.y[index] = 0.0F; vorticity.z[index] = 0.0F; magnitude[index] = smooth_epsilon; return; }
            const float inverse_spacing = 0.5F / grid.cell_size;
            const float wx = (centered_load(centered.z, x, y + 1, z, grid) - centered_load(centered.z, x, y - 1, z, grid) - centered_load(centered.y, x, y, z + 1, grid) + centered_load(centered.y, x, y, z - 1, grid)) * inverse_spacing;
            const float wy = (centered_load(centered.x, x, y, z + 1, grid) - centered_load(centered.x, x, y, z - 1, grid) - centered_load(centered.z, x + 1, y, z, grid) + centered_load(centered.z, x - 1, y, z, grid)) * inverse_spacing;
            const float wz = (centered_load(centered.y, x + 1, y, z, grid) - centered_load(centered.y, x - 1, y, z, grid) - centered_load(centered.x, x, y + 1, z, grid) + centered_load(centered.x, x, y - 1, z, grid)) * inverse_spacing;
            vorticity.x[index] = wx;
            vorticity.y[index] = wy;
            vorticity.z[index] = wz;
            magnitude[index] = sqrtf(wx * wx + wy * wy + wz * wz + smooth_epsilon * smooth_epsilon);
        }

        __global__ void vorticity_normal_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* magnitude, const CenteredVectorView normal, float* normalizer) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (cell_mask[index] != 0u) { normal.x[index] = 0.0F; normal.y[index] = 0.0F; normal.z[index] = 0.0F; normalizer[index] = smooth_epsilon; return; }
            const float inverse_spacing = 0.5F / grid.cell_size;
            const Vector gradient{
                (centered_load(magnitude, x + 1, y, z, grid) - centered_load(magnitude, x - 1, y, z, grid)) * inverse_spacing,
                (centered_load(magnitude, x, y + 1, z, grid) - centered_load(magnitude, x, y - 1, z, grid)) * inverse_spacing,
                (centered_load(magnitude, x, y, z + 1, grid) - centered_load(magnitude, x, y, z - 1, grid)) * inverse_spacing,
            };
            const float length = sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z + smooth_epsilon * smooth_epsilon);
            normal.x[index] = gradient.x / length;
            normal.y[index] = gradient.y / length;
            normal.z[index] = gradient.z / length;
            normalizer[index] = length;
        }

        __global__ void add_vorticity_force_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView normal, const float* confinement, const CenteredVectorView force) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            const float scale = confinement[0] * grid.cell_size;
            force.x[index] += scale * (normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index]);
            force.y[index] += scale * (normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index]);
            force.z[index] += scale * (normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index]);
        }

        __global__ void magnitude_tangent_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView vorticity_tangent, const float* magnitude, float* magnitude_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            magnitude_tangent[index] = cell_mask[index] == 0u ? (vorticity.x[index] * vorticity_tangent.x[index] + vorticity.y[index] * vorticity_tangent.y[index] + vorticity.z[index] * vorticity_tangent.z[index]) / magnitude[index] : 0.0F;
        }

        __global__ void normal_tangent_kernel(const Grid grid, const std::uint32_t* cell_mask, const float* magnitude_tangent, const ConstCenteredVectorView normal, const float* normalizer, const CenteredVectorView normal_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (cell_mask[index] != 0u) { normal_tangent.x[index] = 0.0F; normal_tangent.y[index] = 0.0F; normal_tangent.z[index] = 0.0F; return; }
            const float inverse_spacing = 0.5F / grid.cell_size;
            const Vector gradient_tangent{
                (centered_load(magnitude_tangent, x + 1, y, z, grid) - centered_load(magnitude_tangent, x - 1, y, z, grid)) * inverse_spacing,
                (centered_load(magnitude_tangent, x, y + 1, z, grid) - centered_load(magnitude_tangent, x, y - 1, z, grid)) * inverse_spacing,
                (centered_load(magnitude_tangent, x, y, z + 1, grid) - centered_load(magnitude_tangent, x, y, z - 1, grid)) * inverse_spacing,
            };
            const float projection = normal.x[index] * gradient_tangent.x + normal.y[index] * gradient_tangent.y + normal.z[index] * gradient_tangent.z;
            normal_tangent.x[index] = (gradient_tangent.x - normal.x[index] * projection) / normalizer[index];
            normal_tangent.y[index] = (gradient_tangent.y - normal.y[index] * projection) / normalizer[index];
            normal_tangent.z[index] = (gradient_tangent.z - normal.z[index] * projection) / normalizer[index];
        }

        __global__ void vorticity_force_tangent_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView vorticity_tangent, const ConstCenteredVectorView normal, const ConstCenteredVectorView normal_tangent, const float* confinement, const float* confinement_tangent, const CenteredVectorView force_tangent) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            const Vector cross{
                normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index],
                normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index],
                normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index],
            };
            const Vector cross_tangent{
                normal_tangent.y[index] * vorticity.z[index] + normal.y[index] * vorticity_tangent.z[index] - normal_tangent.z[index] * vorticity.y[index] - normal.z[index] * vorticity_tangent.y[index],
                normal_tangent.z[index] * vorticity.x[index] + normal.z[index] * vorticity_tangent.x[index] - normal_tangent.x[index] * vorticity.z[index] - normal.x[index] * vorticity_tangent.z[index],
                normal_tangent.x[index] * vorticity.y[index] + normal.x[index] * vorticity_tangent.y[index] - normal_tangent.y[index] * vorticity.x[index] - normal.y[index] * vorticity_tangent.x[index],
            };
            force_tangent.x[index] += grid.cell_size * (confinement_tangent[0] * cross.x + confinement[0] * cross_tangent.x);
            force_tangent.y[index] += grid.cell_size * (confinement_tangent[0] * cross.y + confinement[0] * cross_tangent.y);
            force_tangent.z[index] += grid.cell_size * (confinement_tangent[0] * cross.z + confinement[0] * cross_tangent.z);
        }

        __global__ void vorticity_force_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const ConstCenteredVectorView normal, const float* confinement, const ConstCenteredVectorAdjointView force_adjoint, const CenteredVectorAdjointView vorticity_adjoint, const CenteredVectorAdjointView normal_adjoint, double* confinement_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            const Vector cross{
                normal.y[index] * vorticity.z[index] - normal.z[index] * vorticity.y[index],
                normal.z[index] * vorticity.x[index] - normal.x[index] * vorticity.z[index],
                normal.x[index] * vorticity.y[index] - normal.y[index] * vorticity.x[index],
            };
            const double scale = static_cast<double>(confinement[0]) * grid.cell_size;
            atomicAdd(confinement_adjoint, grid.cell_size * (cross.x * force_adjoint.x[index] + cross.y * force_adjoint.y[index] + cross.z * force_adjoint.z[index]));
            normal_adjoint.x[index] += scale * (vorticity.y[index] * force_adjoint.z[index] - vorticity.z[index] * force_adjoint.y[index]);
            normal_adjoint.y[index] += scale * (vorticity.z[index] * force_adjoint.x[index] - vorticity.x[index] * force_adjoint.z[index]);
            normal_adjoint.z[index] += scale * (vorticity.x[index] * force_adjoint.y[index] - vorticity.y[index] * force_adjoint.x[index]);
            vorticity_adjoint.x[index] += scale * (force_adjoint.y[index] * normal.z[index] - force_adjoint.z[index] * normal.y[index]);
            vorticity_adjoint.y[index] += scale * (force_adjoint.z[index] * normal.x[index] - force_adjoint.x[index] * normal.z[index]);
            vorticity_adjoint.z[index] += scale * (force_adjoint.x[index] * normal.y[index] - force_adjoint.y[index] * normal.x[index]);
        }

        __global__ void normal_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView normal, const float* normalizer, const ConstCenteredVectorAdjointView normal_adjoint, double* magnitude_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double projection = normal.x[index] * normal_adjoint.x[index] + normal.y[index] * normal_adjoint.y[index] + normal.z[index] * normal_adjoint.z[index];
            const double gradient_adjoint_x = (normal_adjoint.x[index] - normal.x[index] * projection) / normalizer[index];
            const double gradient_adjoint_y = (normal_adjoint.y[index] - normal.y[index] * projection) / normalizer[index];
            const double gradient_adjoint_z = (normal_adjoint.z[index] - normal.z[index] * projection) / normalizer[index];
            const double scale = 0.5 / grid.cell_size;
            centered_scatter(magnitude_adjoint, x + 1, y, z, grid, scale * gradient_adjoint_x);
            centered_scatter(magnitude_adjoint, x - 1, y, z, grid, -scale * gradient_adjoint_x);
            centered_scatter(magnitude_adjoint, x, y + 1, z, grid, scale * gradient_adjoint_y);
            centered_scatter(magnitude_adjoint, x, y - 1, z, grid, -scale * gradient_adjoint_y);
            centered_scatter(magnitude_adjoint, x, y, z + 1, grid, scale * gradient_adjoint_z);
            centered_scatter(magnitude_adjoint, x, y, z - 1, grid, -scale * gradient_adjoint_z);
        }

        __global__ void magnitude_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorView vorticity, const float* magnitude, const double* magnitude_adjoint, const CenteredVectorAdjointView vorticity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            const double scale = magnitude_adjoint[index] / magnitude[index];
            vorticity_adjoint.x[index] += scale * vorticity.x[index];
            vorticity_adjoint.y[index] += scale * vorticity.y[index];
            vorticity_adjoint.z[index] += scale * vorticity.z[index];
        }

        __global__ void curl_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorAdjointView vorticity_adjoint, const CenteredVectorAdjointView centered_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double scale = 0.5 / grid.cell_size;
            const double ax = vorticity_adjoint.x[index] * scale;
            const double ay = vorticity_adjoint.y[index] * scale;
            const double az = vorticity_adjoint.z[index] * scale;
            centered_scatter(centered_adjoint.z, x, y + 1, z, grid, ax); centered_scatter(centered_adjoint.z, x, y - 1, z, grid, -ax);
            centered_scatter(centered_adjoint.y, x, y, z + 1, grid, -ax); centered_scatter(centered_adjoint.y, x, y, z - 1, grid, ax);
            centered_scatter(centered_adjoint.x, x, y, z + 1, grid, ay); centered_scatter(centered_adjoint.x, x, y, z - 1, grid, -ay);
            centered_scatter(centered_adjoint.z, x + 1, y, z, grid, -ay); centered_scatter(centered_adjoint.z, x - 1, y, z, grid, ay);
            centered_scatter(centered_adjoint.y, x + 1, y, z, grid, az); centered_scatter(centered_adjoint.y, x - 1, y, z, grid, -az);
            centered_scatter(centered_adjoint.x, x, y + 1, z, grid, -az); centered_scatter(centered_adjoint.x, x, y - 1, z, grid, az);
        }

        __global__ void centered_velocity_reverse_kernel(const Grid grid, const std::uint32_t* cell_mask, const ConstCenteredVectorAdjointView centered_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            atomicAdd(velocity_adjoint.x + index3(x, y, z, grid.nx + 1, grid.ny), 0.5F * centered_adjoint.x[index]);
            atomicAdd(velocity_adjoint.x + index3(x + 1, y, z, grid.nx + 1, grid.ny), 0.5F * centered_adjoint.x[index]);
            atomicAdd(velocity_adjoint.y + index3(x, y, z, grid.nx, grid.ny + 1), 0.5F * centered_adjoint.y[index]);
            atomicAdd(velocity_adjoint.y + index3(x, y + 1, z, grid.nx, grid.ny + 1), 0.5F * centered_adjoint.y[index]);
            atomicAdd(velocity_adjoint.z + index3(x, y, z, grid.nx, grid.ny), 0.5F * centered_adjoint.z[index]);
            atomicAdd(velocity_adjoint.z + index3(x, y, z + 1, grid.nx, grid.ny), 0.5F * centered_adjoint.z[index]);
        }

        __device__ bool pressure_periodic(const ScalarBoundaryData boundary, const int dimension) {
            return boundary.modes[dimension * 2] == 2u && boundary.modes[dimension * 2 + 1] == 2u;
        }

        __device__ void pressure_neighbor(const Grid grid, const ScalarBoundaryData boundary, const std::uint32_t* cell_mask, const int x, const int y, const int z, const int dimension, const int direction, int& neighbor_x, int& neighbor_y, int& neighbor_z, bool& connected, bool& fixed, float& fixed_value) {
            neighbor_x = x + (dimension == 0 ? direction : 0);
            neighbor_y = y + (dimension == 1 ? direction : 0);
            neighbor_z = z + (dimension == 2 ? direction : 0);
            const int coordinate = dimension == 0 ? neighbor_x : dimension == 1 ? neighbor_y : neighbor_z;
            const int size = dimension == 0 ? grid.nx : dimension == 1 ? grid.ny : grid.nz;
            if (coordinate < 0 || coordinate >= size) {
                if (pressure_periodic(boundary, dimension)) {
                    if (dimension == 0) neighbor_x = wrap(neighbor_x, grid.nx);
                    if (dimension == 1) neighbor_y = wrap(neighbor_y, grid.ny);
                    if (dimension == 2) neighbor_z = wrap(neighbor_z, grid.nz);
                } else {
                    const int face = dimension * 2 + (direction > 0);
                    fixed = boundary.modes[face] == 0u;
                    fixed_value = boundary.values[face];
                    connected = false;
                    return;
                }
            }
            connected = cell_mask[index3(neighbor_x, neighbor_y, neighbor_z, grid.nx, grid.ny)] == 0u;
            fixed = false;
        }

        __global__ void pressure_rhs_kernel(const Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, float* rhs) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            if (cell_mask[index] != 0u || index == pressure_anchor) { rhs[index] = 0.0F; return; }
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const float divergence = (velocity.x[index3(x + 1, y, z, grid.nx + 1, grid.ny)] - velocity.x[index3(x, y, z, grid.nx + 1, grid.ny)] + velocity.y[index3(x, y + 1, z, grid.nx, grid.ny + 1)] - velocity.y[index3(x, y, z, grid.nx, grid.ny + 1)] + velocity.z[index3(x, y, z + 1, grid.nx, grid.ny)] - velocity.z[index3(x, y, z, grid.nx, grid.ny)]) / grid.cell_size;
            rhs[index] = -grid.cell_size * grid.cell_size * divergence / grid.time_step;
        }

        __global__ void rbgs_kernel(const Grid grid, const int parity, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ScalarBoundaryData boundary, const float* rhs, float* pressure) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (((x + y + z) & 1) != parity || cell_mask[index] != 0u || index == pressure_anchor) return;
            float sum = rhs[index];
            float diagonal = 0.0F;
            for (int dimension = 0; dimension < 3; ++dimension) {
                for (int direction = -1; direction <= 1; direction += 2) {
                    int nx, ny, nz; bool connected, fixed; float fixed_value{};
                    pressure_neighbor(grid, boundary, cell_mask, x, y, z, dimension, direction, nx, ny, nz, connected, fixed, fixed_value);
                    if (connected) { sum += pressure[index3(nx, ny, nz, grid.nx, grid.ny)]; diagonal += 1.0F; }
                    else if (fixed) { sum += fixed_value; diagonal += 1.0F; }
                }
            }
            pressure[index] = diagonal == 0.0F ? 0.0F : sum / diagonal;
        }

        __device__ bool projectable_face(const Grid grid, const int axis, const int x, const int y, const int z, const std::uint32_t* cell_mask, int& first_x, int& first_y, int& first_z, int& second_x, int& second_y, int& second_z) {
            first_x = x - (axis == 0); first_y = y - (axis == 1); first_z = z - (axis == 2);
            second_x = x; second_y = y; second_z = z;
            if (first_x < 0 || first_x >= grid.nx || first_y < 0 || first_y >= grid.ny || first_z < 0 || first_z >= grid.nz) return false;
            if (second_x < 0 || second_x >= grid.nx || second_y < 0 || second_y >= grid.ny || second_z < 0 || second_z >= grid.nz) return false;
            return cell_mask[index3(first_x, first_y, first_z, grid.nx, grid.ny)] == 0u && cell_mask[index3(second_x, second_y, second_z, grid.nx, grid.ny)] == 0u;
        }

        __global__ void project_velocity_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const float* velocity, const float* pressure, float* output) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            int fx, fy, fz, sx, sy, sz;
            if (!projectable_face(grid, axis, x, y, z, cell_mask, fx, fy, fz, sx, sy, sz)) { output[index] = velocity[index]; return; }
            output[index] = velocity[index] - grid.time_step * (pressure[index3(sx, sy, sz, grid.nx, grid.ny)] - pressure[index3(fx, fy, fz, grid.nx, grid.ny)]) / grid.cell_size;
        }

        __global__ void project_velocity_reverse_kernel(const Grid grid, const int axis, const std::uint32_t* cell_mask, const double* output_adjoint, double* velocity_adjoint, double* pressure_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= face_count(grid, axis)) return;
            velocity_adjoint[index] += output_adjoint[index];
            int x, y, z;
            decode(index, extent(grid, axis, 0), extent(grid, axis, 1), x, y, z);
            int fx, fy, fz, sx, sy, sz;
            if (!projectable_face(grid, axis, x, y, z, cell_mask, fx, fy, fz, sx, sy, sz)) return;
            const double scale = static_cast<double>(grid.time_step) * output_adjoint[index] / grid.cell_size;
            atomicAdd(pressure_adjoint + index3(fx, fy, fz, grid.nx, grid.ny), scale);
            atomicAdd(pressure_adjoint + index3(sx, sy, sz, grid.nx, grid.ny), -scale);
        }

        __global__ void rbgs_reverse_kernel(const Grid grid, const int parity, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ScalarBoundaryData boundary, double* pressure_adjoint, double* rhs_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid)) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            if (((x + y + z) & 1) != parity || cell_mask[index] != 0u || index == pressure_anchor) return;
            float diagonal = 0.0F;
            int neighbors[6]; int neighbor_count = 0;
            for (int dimension = 0; dimension < 3; ++dimension) {
                for (int direction = -1; direction <= 1; direction += 2) {
                    int nx, ny, nz; bool connected, fixed; float fixed_value{};
                    pressure_neighbor(grid, boundary, cell_mask, x, y, z, dimension, direction, nx, ny, nz, connected, fixed, fixed_value);
                    if (connected) { neighbors[neighbor_count++] = static_cast<int>(index3(nx, ny, nz, grid.nx, grid.ny)); diagonal += 1.0F; }
                    else if (fixed) diagonal += 1.0F;
                }
            }
            const double adjoint = diagonal == 0.0F ? 0.0 : pressure_adjoint[index] / diagonal;
            pressure_adjoint[index] = 0.0;
            rhs_adjoint[index] += adjoint;
            for (int neighbor = 0; neighbor < neighbor_count; ++neighbor) atomicAdd(pressure_adjoint + neighbors[neighbor], adjoint);
        }

        __global__ void pressure_rhs_reverse_kernel(const Grid grid, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const double* rhs_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
            const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= cell_count(grid) || cell_mask[index] != 0u || index == pressure_anchor) return;
            int x, y, z;
            decode(index, grid.nx, grid.ny, x, y, z);
            const double scale = -static_cast<double>(grid.cell_size) * rhs_adjoint[index] / grid.time_step;
            atomicAdd(velocity_adjoint.x + index3(x + 1, y, z, grid.nx + 1, grid.ny), scale);
            atomicAdd(velocity_adjoint.x + index3(x, y, z, grid.nx + 1, grid.ny), -scale);
            atomicAdd(velocity_adjoint.y + index3(x, y + 1, z, grid.nx, grid.ny + 1), scale);
            atomicAdd(velocity_adjoint.y + index3(x, y, z, grid.nx, grid.ny + 1), -scale);
            atomicAdd(velocity_adjoint.z + index3(x, y, z + 1, grid.nx, grid.ny), scale);
            atomicAdd(velocity_adjoint.z + index3(x, y, z, grid.nx, grid.ny), -scale);
        }

    } // namespace

    void vorticity_forward(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const float* confinement, const VorticityView cache, const CenteredVectorView force) {
        centered_velocity_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, velocity, cache.centered_velocity);
        curl_magnitude_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, {cache.centered_velocity.x, cache.centered_velocity.y, cache.centered_velocity.z}, cache.vorticity, cache.magnitude.values);
        vorticity_normal_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, cache.magnitude.values, cache.normal, cache.normalizer.values);
        add_vorticity_force_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, {cache.vorticity.x, cache.vorticity.y, cache.vorticity.z}, {cache.normal.x, cache.normal.y, cache.normal.z}, confinement, force);
        check_launch("vorticity_forward");
    }

    void vorticity_jvp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity_tangent, const float* confinement, const float* confinement_tangent, const ConstVorticityView cache, const CenteredVectorView force_tangent, const VorticityTangentScratch tangent_scratch) {
        centered_velocity_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, velocity_tangent, tangent_scratch.centered_velocity);
        curl_magnitude_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, {tangent_scratch.centered_velocity.x, tangent_scratch.centered_velocity.y, tangent_scratch.centered_velocity.z}, tangent_scratch.vorticity, tangent_scratch.magnitude.values);
        magnitude_tangent_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, cache.vorticity, {tangent_scratch.vorticity.x, tangent_scratch.vorticity.y, tangent_scratch.vorticity.z}, cache.magnitude.values, tangent_scratch.magnitude.values);
        normal_tangent_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, tangent_scratch.magnitude.values, cache.normal, cache.normalizer.values, tangent_scratch.normal);
        vorticity_force_tangent_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, cache.vorticity, {tangent_scratch.vorticity.x, tangent_scratch.vorticity.y, tangent_scratch.vorticity.z}, cache.normal, {tangent_scratch.normal.x, tangent_scratch.normal.y, tangent_scratch.normal.z}, confinement, confinement_tangent, force_tangent);
        check_launch("vorticity_jvp");
    }

    void vorticity_vjp(const cudaStream_t stream, const Grid grid, const std::uint32_t* cell_mask, const float* confinement, const ConstVorticityView cache, const ConstCenteredVectorAdjointView force_adjoint, const StaggeredVectorAdjointView velocity_adjoint, double* confinement_adjoint, const VorticityAdjointScratch scratch) {
        const std::size_t bytes = static_cast<std::size_t>(cell_count(grid)) * sizeof(double);
        cudaMemsetAsync(scratch.centered_velocity.x, 0, bytes, stream); cudaMemsetAsync(scratch.centered_velocity.y, 0, bytes, stream); cudaMemsetAsync(scratch.centered_velocity.z, 0, bytes, stream);
        cudaMemsetAsync(scratch.vorticity.x, 0, bytes, stream); cudaMemsetAsync(scratch.vorticity.y, 0, bytes, stream); cudaMemsetAsync(scratch.vorticity.z, 0, bytes, stream);
        cudaMemsetAsync(scratch.magnitude.values, 0, bytes, stream);
        cudaMemsetAsync(scratch.normal.x, 0, bytes, stream); cudaMemsetAsync(scratch.normal.y, 0, bytes, stream); cudaMemsetAsync(scratch.normal.z, 0, bytes, stream);
        vorticity_force_reverse_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, cache.vorticity, cache.normal, confinement, force_adjoint, scratch.vorticity, scratch.normal, confinement_adjoint);
        normal_reverse_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, cache.normal, cache.normalizer.values, {scratch.normal.x, scratch.normal.y, scratch.normal.z}, scratch.magnitude.values);
        magnitude_reverse_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, cache.vorticity, cache.magnitude.values, scratch.magnitude.values, scratch.vorticity);
        curl_reverse_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, {scratch.vorticity.x, scratch.vorticity.y, scratch.vorticity.z}, scratch.centered_velocity);
        centered_velocity_reverse_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, cell_mask, {scratch.centered_velocity.x, scratch.centered_velocity.y, scratch.centered_velocity.z}, velocity_adjoint);
        check_launch("vorticity_vjp");
    }

    void projection_forward(const cudaStream_t stream, const Grid grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity, const VelocityBoundaryData, const ScalarBoundaryData pressure_boundary, const ScalarView pressure, const ScalarView rhs, const StaggeredVectorView output) {
        cudaMemsetAsync(pressure.values, 0, static_cast<std::size_t>(cell_count(grid)) * sizeof(float), stream);
        pressure_rhs_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, pressure_anchor, cell_mask, velocity, rhs.values);
        for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
            rbgs_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, 0, pressure_anchor, cell_mask, pressure_boundary, rhs.values, pressure.values);
            rbgs_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, 1, pressure_anchor, cell_mask, pressure_boundary, rhs.values, pressure.values);
        }
        for (int axis = 0; axis < 3; ++axis) project_velocity_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, component(velocity, axis), pressure.values, component(output, axis));
        check_launch("projection_forward");
    }

    void projection_jvp(const cudaStream_t stream, const Grid grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ConstStaggeredVectorView velocity_tangent, const VelocityBoundaryData velocity_boundary, ScalarBoundaryData pressure_boundary, const ScalarView pressure_tangent, const ScalarView rhs_tangent, const StaggeredVectorView output_tangent) {
        for (float& value : pressure_boundary.values) value = 0.0F;
        projection_forward(stream, grid, iterations, pressure_anchor, cell_mask, velocity_tangent, velocity_boundary, pressure_boundary, pressure_tangent, rhs_tangent, output_tangent);
    }

    void projection_vjp(const cudaStream_t stream, const Grid grid, const std::uint32_t iterations, const std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, const ConstStaggeredVectorAdjointView output_adjoint, const VelocityBoundaryData, const ScalarBoundaryData pressure_boundary, const ScalarAdjointView pressure_adjoint, const ScalarAdjointView rhs_adjoint, const StaggeredVectorAdjointView velocity_adjoint) {
        const std::size_t bytes = static_cast<std::size_t>(cell_count(grid)) * sizeof(double);
        cudaMemsetAsync(pressure_adjoint.values, 0, bytes, stream);
        cudaMemsetAsync(rhs_adjoint.values, 0, bytes, stream);
        for (int axis = 0; axis < 3; ++axis) project_velocity_reverse_kernel<<<blocks(face_count(grid, axis)), block_size, 0, stream>>>(grid, axis, cell_mask, component(output_adjoint, axis), component(velocity_adjoint, axis), pressure_adjoint.values);
        for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
            rbgs_reverse_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, 1, pressure_anchor, cell_mask, pressure_boundary, pressure_adjoint.values, rhs_adjoint.values);
            rbgs_reverse_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, 0, pressure_anchor, cell_mask, pressure_boundary, pressure_adjoint.values, rhs_adjoint.values);
        }
        pressure_rhs_reverse_kernel<<<blocks(cell_count(grid)), block_size, 0, stream>>>(grid, pressure_anchor, cell_mask, rhs_adjoint.values, velocity_adjoint);
        check_launch("projection_vjp");
    }

} // namespace xayah::smoke::cuda_kernels
