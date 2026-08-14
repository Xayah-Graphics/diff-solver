#include "simulation.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace xayah::cloth::examples::forward::simulation_cuda {

    namespace {

        constexpr std::uint32_t block_size = 256u;
        constexpr float two_pi             = 6.28318530717958647692F;
        constexpr float one_third_pi       = 1.04719755119659774615F;

        struct WindVelocity {
            float x;
            float z;
        };

        __device__ double atomic_max(double* const address, const double value) {
            unsigned long long* const integer_address = reinterpret_cast<unsigned long long*>(address);
            unsigned long long previous               = *integer_address;
            while (__longlong_as_double(previous) < value) {
                const unsigned long long assumed = previous;
                previous                         = atomicCAS(integer_address, assumed, __double_as_longlong(value));
                if (previous == assumed) break;
            }
            return __longlong_as_double(previous);
        }

        __device__ WindVelocity wind_velocity_at(const float time, const float normalized_x, const float normalized_y, const float wind_speed, const float gust_strength, const float gust_frequency, const float ramp_duration) {
            const float ramp_coordinate = fminf(time / ramp_duration, 1.0F);
            const float ramp            = ramp_coordinate * ramp_coordinate * (3.0F - 2.0F * ramp_coordinate);
            const float primary_gust    = sinf(two_pi * (gust_frequency * time - 0.85F * normalized_x + 0.12F * normalized_y));
            const float secondary_gust  = sinf(two_pi * (1.73F * gust_frequency * time - 1.70F * normalized_x - 0.28F * normalized_y) + one_third_pi);
            return {
                .x = ramp * wind_speed * (1.0F + 0.20F * gust_strength * primary_gust),
                .z = ramp * wind_speed * gust_strength * (0.72F * primary_gust + 0.28F * secondary_gust),
            };
        }

        __global__ void write_control(const std::uint32_t rows, const std::uint32_t columns, const std::uint64_t step, const float time_step, const float width, const float height, const float wind_speed, const float gust_strength, const float gust_frequency, const float air_density, const float drag_coefficient, const float skin_drag_coefficient, const float ramp_duration, const cuda_kernel::ConstField positions, const cuda_kernel::ConstField velocities, const cuda_kernel::Field external_forces, double* const metrics) {
            const std::uint32_t particle_count = rows * columns;
            const std::uint32_t particle       = blockIdx.x * blockDim.x + threadIdx.x;
            if (particle >= particle_count) return;
            const std::uint32_t row    = particle / columns;
            const std::uint32_t column = particle % columns;
            const float normalized_x   = static_cast<float>(column) / static_cast<float>(columns - 1u);
            const float normalized_y   = static_cast<float>(row) / static_cast<float>(rows - 1u);
            const float time           = static_cast<float>(step) * time_step;
            if (particle == 0u) {
                const WindVelocity first  = wind_velocity_at(time, 0.25F, 0.5F, wind_speed, gust_strength, gust_frequency, ramp_duration);
                const WindVelocity second = wind_velocity_at(time, 0.50F, 0.5F, wind_speed, gust_strength, gust_frequency, ramp_duration);
                const WindVelocity third  = wind_velocity_at(time, 0.75F, 0.5F, wind_speed, gust_strength, gust_frequency, ramp_duration);
                metrics[7]                = first.x;
                metrics[8]                = first.z;
                metrics[9]                = second.x;
                metrics[10]               = second.z;
                metrics[11]               = third.x;
                metrics[12]               = third.z;
            }
            if (column == 0u) {
                external_forces.x[particle] = 0.0F;
                external_forces.y[particle] = 0.0F;
                external_forces.z[particle] = 0.0F;
                return;
            }

            const std::uint32_t left          = row * columns + column - 1u;
            const std::uint32_t right         = row * columns + (column + 1u < columns ? column + 1u : columns - 1u);
            const std::uint32_t top           = (row == 0u ? 0u : row - 1u) * columns + column;
            const std::uint32_t bottom        = (row + 1u < rows ? row + 1u : rows - 1u) * columns + column;
            const float tangent_x_x           = positions.x[right] - positions.x[left];
            const float tangent_x_y           = positions.y[right] - positions.y[left];
            const float tangent_x_z           = positions.z[right] - positions.z[left];
            const float tangent_y_x           = positions.x[bottom] - positions.x[top];
            const float tangent_y_y           = positions.y[bottom] - positions.y[top];
            const float tangent_y_z           = positions.z[bottom] - positions.z[top];
            float normal_x                    = tangent_x_y * tangent_y_z - tangent_x_z * tangent_y_y;
            float normal_y                    = tangent_x_z * tangent_y_x - tangent_x_x * tangent_y_z;
            float normal_z                    = tangent_x_x * tangent_y_y - tangent_x_y * tangent_y_x;
            const float inverse_normal_length = rsqrtf(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
            normal_x *= inverse_normal_length;
            normal_y *= inverse_normal_length;
            normal_z *= inverse_normal_length;

            const WindVelocity local_wind = wind_velocity_at(time, normalized_x, normalized_y, wind_speed, gust_strength, gust_frequency, ramp_duration);
            const float relative_x        = local_wind.x - velocities.x[particle];
            const float relative_y        = -velocities.y[particle];
            const float relative_z        = local_wind.z - velocities.z[particle];
            const float normal_speed      = relative_x * normal_x + relative_y * normal_y + relative_z * normal_z;
            const float tangent_x         = relative_x - normal_speed * normal_x;
            const float tangent_y         = relative_y - normal_speed * normal_y;
            const float tangent_z         = relative_z - normal_speed * normal_z;
            const float tangent_speed     = sqrtf(tangent_x * tangent_x + tangent_y * tangent_y + tangent_z * tangent_z);
            const float edge_weight_x     = column == columns - 1u ? 0.5F : 1.0F;
            const float edge_weight_y     = row == 0u || row == rows - 1u ? 0.5F : 1.0F;
            const float particle_area     = width * height / static_cast<float>((columns - 1u) * (rows - 1u));
            const float area_scale        = 0.5F * air_density * particle_area * edge_weight_x * edge_weight_y;
            const float normal_force      = area_scale * drag_coefficient * normal_speed * fabsf(normal_speed);
            const float tangent_force     = area_scale * skin_drag_coefficient * tangent_speed;
            const float force_x           = normal_force * normal_x + tangent_force * tangent_x;
            const float force_y           = normal_force * normal_y + tangent_force * tangent_y;
            const float force_z           = normal_force * normal_z + tangent_force * tangent_z;
            external_forces.x[particle]   = force_x;
            external_forces.y[particle]   = force_y;
            external_forces.z[particle]   = force_z;
            atomicAdd(metrics + 13u, static_cast<double>(sqrtf(force_x * force_x + force_y * force_y + force_z * force_z)));
        }

        __global__ void reduce_particles(const std::uint32_t particle_count, const std::uint32_t columns, const float* const masses, const cuda_kernel::ConstField positions, const cuda_kernel::ConstField velocities, double* const metrics) {
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

        __global__ void reduce_strain(const std::uint32_t spring_count, const std::uint32_t* const first, const std::uint32_t* const second, const float* const rest_lengths, const cuda_kernel::ConstField positions, double* const maximum_absolute_strain) {
            const std::uint32_t spring = blockIdx.x * blockDim.x + threadIdx.x;
            if (spring >= spring_count) return;
            const std::uint32_t first_particle  = first[spring];
            const std::uint32_t second_particle = second[spring];
            const double dx                     = static_cast<double>(positions.x[second_particle]) - positions.x[first_particle];
            const double dy                     = static_cast<double>(positions.y[second_particle]) - positions.y[first_particle];
            const double dz                     = static_cast<double>(positions.z[second_particle]) - positions.z[first_particle];
            const double strain                 = fabs(sqrt(dx * dx + dy * dy + dz * dz) / rest_lengths[spring] - 1.0);
            atomic_max(maximum_absolute_strain, strain);
        }

        std::uint32_t blocks(const std::uint32_t count) {
            return (count + block_size - 1u) / block_size;
        }

        void check_launch() {
            if (const cudaError_t result = cudaGetLastError(); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        }

    } // namespace

    void launch_write_control(const cudaStream_t stream, const std::uint32_t rows, const std::uint32_t columns, const std::uint64_t step, const float time_step, const float width, const float height, const float wind_speed, const float gust_strength, const float gust_frequency, const float air_density, const float drag_coefficient, const float skin_drag_coefficient, const float ramp_duration, const cuda_kernel::ConstField positions, const cuda_kernel::ConstField velocities, const cuda_kernel::Field external_forces, double* const metrics) {
        write_control<<<blocks(rows * columns), block_size, 0u, stream>>>(rows, columns, step, time_step, width, height, wind_speed, gust_strength, gust_frequency, air_density, drag_coefficient, skin_drag_coefficient, ramp_duration, positions, velocities, external_forces, metrics);
        check_launch();
    }

    void launch_particle_metrics(const cudaStream_t stream, const std::uint32_t rows, const std::uint32_t columns, const float* const masses, const cuda_kernel::ConstField positions, const cuda_kernel::ConstField velocities, double* const metrics) {
        const std::uint32_t particle_count = rows * columns;
        reduce_particles<<<blocks(particle_count), block_size, 0u, stream>>>(particle_count, columns, masses, positions, velocities, metrics);
        check_launch();
    }

    void launch_strain_metrics(const cudaStream_t stream, const std::uint32_t spring_count, const std::uint32_t* const first, const std::uint32_t* const second, const float* const rest_lengths, const cuda_kernel::ConstField positions, double* const maximum_absolute_strain) {
        if (spring_count == 0u) return;
        reduce_strain<<<blocks(spring_count), block_size, 0u, stream>>>(spring_count, first, second, rest_lengths, positions, maximum_absolute_strain);
        check_launch();
    }

} // namespace xayah::cloth::examples::forward::simulation_cuda
