#ifndef XAYAH_EXAMPLES_CLOTH_FORWARD_SIMULATION_H
#define XAYAH_EXAMPLES_CLOTH_FORWARD_SIMULATION_H

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cloth/cuda_types.h>

namespace xayah::cloth::examples::forward::simulation_cuda {
    void launch_write_control(cudaStream_t stream, std::uint32_t rows, std::uint32_t columns, std::uint64_t step, float time_step, float width, float height, float wind_speed, float gust_strength, float gust_frequency, float air_density, float drag_coefficient, float skin_drag_coefficient, float ramp_duration, cuda_kernel::ConstField positions, cuda_kernel::ConstField velocities, cuda_kernel::Field external_forces, double* metrics);

    void launch_particle_metrics(cudaStream_t stream, std::uint32_t rows, std::uint32_t columns, const float* masses, cuda_kernel::ConstField positions, cuda_kernel::ConstField velocities, double* metrics);

    void launch_strain_metrics(cudaStream_t stream, std::uint32_t spring_count, const std::uint32_t* first, const std::uint32_t* second, const float* rest_lengths, cuda_kernel::ConstField positions, double* maximum_absolute_strain);

} // namespace xayah::cloth::examples::forward::simulation_cuda

#endif
