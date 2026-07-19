#ifndef XAYAH_EXAMPLES_CLOTH_FORWARD_SIMULATION_H
#define XAYAH_EXAMPLES_CLOTH_FORWARD_SIMULATION_H

#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::cloth::examples::forward::simulation_cuda {

    struct Field {
        float* x;
        float* y;
        float* z;
    };

    struct ConstField {
        const float* x;
        const float* y;
        const float* z;
    };

    void launch_write_control(cudaStream_t stream, std::uint32_t rows, std::uint32_t columns, std::uint64_t step, float time_step, float mass, float ramp_duration, float period, float base_acceleration, float primary_acceleration, float secondary_acceleration, Field external_forces, double* metrics);

    void launch_particle_metrics(cudaStream_t stream, std::uint32_t rows, std::uint32_t columns, const float* masses, ConstField positions, ConstField velocities, double* metrics);

    void launch_strain_metrics(cudaStream_t stream, std::uint32_t spring_count, const std::uint32_t* first, const std::uint32_t* second, const float* rest_lengths, ConstField positions, double* maximum_absolute_strain);

} // namespace xayah::cloth::examples::forward::simulation_cuda

#endif
