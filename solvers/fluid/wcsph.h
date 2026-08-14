#ifndef XAYAH_FLUID_WCSPH_H
#define XAYAH_FLUID_WCSPH_H

#include "sph.h"

#include <cstdint>

namespace xayah::fluid::wcsph::cuda_kernel {

    void launch_external_forward(void* stream, std::uint32_t particle_count, float gravity_x, float gravity_y, float gravity_z, sph::cuda_kernel::ConstVector controls, sph::cuda_kernel::Vector accelerations);

    void launch_eos_forward(void* stream, std::uint32_t particle_count, const float* densities, sph::cuda_kernel::ParticleParameters particles, const float* speed_of_sound, const float* tait_exponent, float* pressures);
    void launch_eos_jvp(void* stream, std::uint32_t particle_count, const float* densities, const float* density_tangent, sph::cuda_kernel::ParticleParameters particles, sph::cuda_kernel::ParticleParameterTangent particle_tangent, const float* speed_of_sound, const float* speed_of_sound_tangent, const float* tait_exponent, const float* tait_exponent_tangent, float* pressure_tangent);
    void launch_eos_vjp(void* stream, std::uint32_t particle_count, const float* densities, sph::cuda_kernel::ParticleParameters particles, const float* speed_of_sound, const float* tait_exponent, const double* pressure_adjoint, double* density_adjoint, sph::cuda_kernel::ParticleParameterAdjoint particle_adjoint, double* speed_of_sound_adjoint, double* tait_exponent_adjoint);

    void launch_artificial_viscosity_forward(void* stream, std::uint32_t particle_count, float support_radius, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ConstVector velocities, sph::cuda_kernel::ParticleParameters particles, const float* speed_of_sound, sph::cuda_kernel::Neighborhood neighborhood, sph::cuda_kernel::Boundary boundary, const float* densities, sph::cuda_kernel::Vector accelerations);
    void launch_artificial_viscosity_jvp(void* stream, std::uint32_t particle_count, float support_radius, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ConstVector velocities, sph::cuda_kernel::ConstVector position_tangent, sph::cuda_kernel::ConstVector velocity_tangent, sph::cuda_kernel::ParticleParameters particles, sph::cuda_kernel::ParticleParameterTangent particle_tangent, const float* speed_of_sound, const float* speed_of_sound_tangent, sph::cuda_kernel::Neighborhood neighborhood, sph::cuda_kernel::Boundary boundary, const float* densities, const float* density_tangent, sph::cuda_kernel::Vector acceleration_tangent);
    void launch_artificial_viscosity_vjp(void* stream, std::uint32_t particle_count, float support_radius, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ConstVector velocities, sph::cuda_kernel::ParticleParameters particles, const float* speed_of_sound, sph::cuda_kernel::Neighborhood neighborhood, sph::cuda_kernel::Boundary boundary, const float* densities, sph::cuda_kernel::ConstVectorAdjoint acceleration_adjoint, sph::cuda_kernel::VectorAdjoint position_adjoint, sph::cuda_kernel::VectorAdjoint velocity_adjoint, double* density_adjoint, sph::cuda_kernel::ParticleParameterAdjoint particle_adjoint, double* speed_of_sound_adjoint);

    void launch_surface_forward(void* stream, std::uint32_t particle_count, float support_radius, float particle_radius, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ParticleParameters particles, const float* boundary_surface_tension, sph::cuda_kernel::Neighborhood neighborhood, sph::cuda_kernel::Boundary boundary, sph::cuda_kernel::Vector accelerations);
    void launch_surface_jvp(void* stream, std::uint32_t particle_count, float support_radius, float particle_radius, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ConstVector position_tangent, sph::cuda_kernel::ParticleParameters particles, sph::cuda_kernel::ParticleParameterTangent particle_tangent, const float* boundary_surface_tension, const float* boundary_surface_tension_tangent, sph::cuda_kernel::Neighborhood neighborhood, sph::cuda_kernel::Boundary boundary, sph::cuda_kernel::Vector acceleration_tangent);
    void launch_surface_vjp(void* stream, std::uint32_t particle_count, float support_radius, float particle_radius, sph::cuda_kernel::ConstVector positions, sph::cuda_kernel::ParticleParameters particles, const float* boundary_surface_tension, sph::cuda_kernel::Neighborhood neighborhood, sph::cuda_kernel::Boundary boundary, sph::cuda_kernel::ConstVectorAdjoint acceleration_adjoint, sph::cuda_kernel::VectorAdjoint position_adjoint, sph::cuda_kernel::ParticleParameterAdjoint particle_adjoint, double* boundary_surface_tension_adjoint);

} // namespace xayah::fluid::wcsph::cuda_kernel

#endif
