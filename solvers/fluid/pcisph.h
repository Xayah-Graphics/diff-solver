#ifndef XAYAH_FLUID_PCISPH_H
#define XAYAH_FLUID_PCISPH_H

#include "sph.h"

#include <cstdint>

namespace xayah::fluid::pcisph::cuda_kernel {

    void launch_predict_forward(void* stream, std::uint32_t particle_count, float time_step, sph::cuda_kernel::Domain domain, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ConstVector velocities, sph::cuda_kernel::ConstVector non_pressure_accelerations, sph::cuda_kernel::ConstVector pressure_accelerations, sph::cuda_kernel::Vector predicted_positions, sph::cuda_kernel::Vector predicted_velocities);
    void launch_predict_jvp(void* stream, std::uint32_t particle_count, float time_step, sph::cuda_kernel::Domain domain, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ConstVector velocities, sph::cuda_kernel::ConstVector non_pressure_accelerations, sph::cuda_kernel::ConstVector pressure_accelerations, sph::cuda_kernel::ConstVector position_tangent, sph::cuda_kernel::ConstVector velocity_tangent, sph::cuda_kernel::ConstVector non_pressure_acceleration_tangent, sph::cuda_kernel::ConstVector pressure_acceleration_tangent, sph::cuda_kernel::Vector predicted_position_tangent, sph::cuda_kernel::Vector predicted_velocity_tangent);
    void launch_predict_vjp(void* stream, std::uint32_t particle_count, float time_step, sph::cuda_kernel::Domain domain, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ConstVector velocities, sph::cuda_kernel::ConstVector non_pressure_accelerations, sph::cuda_kernel::ConstVector pressure_accelerations, sph::cuda_kernel::ConstVectorAdjoint predicted_position_adjoint, sph::cuda_kernel::ConstVectorAdjoint predicted_velocity_adjoint, sph::cuda_kernel::VectorAdjoint position_adjoint, sph::cuda_kernel::VectorAdjoint velocity_adjoint, sph::cuda_kernel::VectorAdjoint non_pressure_acceleration_adjoint, sph::cuda_kernel::VectorAdjoint pressure_acceleration_adjoint);

    void launch_pressure_update_forward(void* stream, std::uint32_t particle_count, float time_step, float reference_gradient_norm, sph::cuda_kernel::ParticleParameters particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, float* pressures);
    void launch_pressure_update_jvp(void* stream, std::uint32_t particle_count, float time_step, float reference_gradient_norm, sph::cuda_kernel::ParticleParameters particles, sph::cuda_kernel::ParticleParameterTangent particle_tangent, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const float* previous_pressure_tangent, const float* predicted_density_tangent, const float* pressure_relaxation_tangent, float* pressure_tangent);
    void launch_pressure_update_vjp(void* stream, std::uint32_t particle_count, float time_step, float reference_gradient_norm, sph::cuda_kernel::ParticleParameters particles, const float* previous_pressures, const float* predicted_densities, const float* pressure_relaxation, const double* pressure_adjoint, sph::cuda_kernel::ParticleParameterAdjoint particle_adjoint, double* previous_pressure_adjoint, double* predicted_density_adjoint, double* pressure_relaxation_adjoint);

    void launch_copy_iteration_forward(void* stream, std::uint32_t particle_count, const float* source_pressures, const float* source_densities, sph::cuda_kernel::ConstVector source_accelerations, sph::cuda_kernel::ConstVector source_positions, sph::cuda_kernel::ConstVector source_velocities, float* destination_pressures, float* destination_densities, sph::cuda_kernel::Vector destination_accelerations, sph::cuda_kernel::Vector destination_positions, sph::cuda_kernel::Vector destination_velocities);

} // namespace xayah::fluid::pcisph::cuda_kernel

#endif
