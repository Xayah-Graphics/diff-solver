#ifndef XAYAH_FLUID_SPH_H
#define XAYAH_FLUID_SPH_H

#include <cstddef>
#include <cstdint>

namespace xayah::fluid::sph::cuda_kernel {

    struct ConstVector {
        const float* x;
        const float* y;
        const float* z;
    };

    struct Vector {
        float* x;
        float* y;
        float* z;

        operator ConstVector() const { return {.x = x, .y = y, .z = z}; }
    };

    struct ConstVectorAdjoint {
        const double* x;
        const double* y;
        const double* z;
    };

    struct VectorAdjoint {
        double* x;
        double* y;
        double* z;

        operator ConstVectorAdjoint() const { return {.x = x, .y = y, .z = z}; }
    };

    struct Neighborhood {
        const std::uint64_t* sorted_keys;
        const std::uint32_t* sorted_particle_indices;
        const std::uint32_t* cell_offsets;
        const std::uint64_t* sorted_boundary_keys;
        const std::uint32_t* sorted_boundary_indices;
        const std::uint32_t* boundary_cell_offsets;
        std::uint32_t particle_count;
        std::uint32_t boundary_count;
        std::uint32_t cells_x;
        std::uint32_t cells_y;
        std::uint32_t cells_z;
        float origin_x;
        float origin_y;
        float origin_z;
        float cell_size;
    };

    struct Boundary {
        const float* position_x;
        const float* position_y;
        const float* position_z;
        const float* velocity_x;
        const float* velocity_y;
        const float* velocity_z;
        const float* volumes;
        std::uint32_t count;
        float time;
    };

    struct Domain {
        float minimum_x;
        float minimum_y;
        float minimum_z;
        float maximum_x;
        float maximum_y;
        float maximum_z;
        float velocity_x;
        float velocity_y;
        float velocity_z;
        std::uint32_t no_slip;
    };

    struct ParticleParameters {
        const float* masses;
        const float* rest_densities;
        const float* viscosities;
        const float* surface_tensions;
    };

    struct ParticleParameterTangent {
        const float* masses;
        const float* rest_densities;
        const float* viscosities;
        const float* surface_tensions;
    };

    struct ParticleParameterAdjoint {
        double* masses;
        double* rest_densities;
        double* viscosities;
        double* surface_tensions;
    };

    void launch_copy_vector(void* stream, std::uint32_t count, ConstVector source, Vector destination);
    void launch_copy_vector_adjoint(void* stream, std::uint32_t count, ConstVectorAdjoint source, VectorAdjoint destination);
    void launch_accumulate_vector_adjoint(void* stream, std::uint32_t count, ConstVectorAdjoint source, VectorAdjoint destination);
    void launch_copy_scalar(void* stream, std::uint32_t count, const float* source, float* destination);
    void launch_copy_scalar_adjoint(void* stream, std::uint32_t count, const double* source, double* destination);
    void launch_accumulate_scalar_adjoint(void* stream, std::uint32_t count, const double* source, double* destination);

    void query_neighbor_storage(std::uint32_t particle_count, std::size_t& sort_bytes);
    void launch_build_neighborhood(void* stream, std::uint32_t particle_count, std::uint32_t boundary_count, float support_radius, float time_step, std::uint64_t step_index, Domain domain, ConstVector positions, ConstVector boundary_positions, ConstVector boundary_velocities, std::uint64_t* unsorted_keys, std::uint32_t* unsorted_particle_indices, std::uint64_t* unsorted_boundary_keys, std::uint32_t* unsorted_boundary_indices, void* sort_scratch, std::size_t sort_scratch_bytes, void* boundary_sort_scratch, std::size_t boundary_sort_scratch_bytes, std::uint64_t* sorted_keys, std::uint32_t* sorted_particle_indices, std::uint32_t* cell_offsets, std::uint64_t* sorted_boundary_keys, std::uint32_t* sorted_boundary_indices, std::uint32_t* boundary_cell_offsets);

    void launch_density_forward(void* stream, std::uint32_t particle_count, float support_radius, ConstVector topology_positions, ConstVector positions, ParticleParameters parameters, Neighborhood neighborhood, Boundary boundary, float* densities);
    void launch_density_jvp(void* stream, std::uint32_t particle_count, float support_radius, ConstVector topology_positions, ConstVector positions, ConstVector position_tangent, ParticleParameters parameters, ParticleParameterTangent parameter_tangent, Neighborhood neighborhood, Boundary boundary, float* density_tangent);
    void launch_density_vjp(void* stream, std::uint32_t particle_count, float support_radius, ConstVector topology_positions, ConstVector positions, ParticleParameters parameters, Neighborhood neighborhood, Boundary boundary, const double* density_adjoint, VectorAdjoint position_adjoint, ParticleParameterAdjoint parameter_adjoint);
    void launch_pbf_density_forward(void* stream, std::uint32_t particle_count, float support_radius, ConstVector topology_positions, ConstVector positions, ParticleParameters parameters, Neighborhood neighborhood, Boundary boundary, float* densities);
    void launch_pbf_density_jvp(void* stream, std::uint32_t particle_count, float support_radius, ConstVector topology_positions, ConstVector positions, ConstVector position_tangent, ParticleParameters parameters, ParticleParameterTangent parameter_tangent, Neighborhood neighborhood, Boundary boundary, float* density_tangent);
    void launch_pbf_density_vjp(void* stream, std::uint32_t particle_count, float support_radius, ConstVector topology_positions, ConstVector positions, ParticleParameters parameters, Neighborhood neighborhood, Boundary boundary, const double* density_adjoint, VectorAdjoint position_adjoint, ParticleParameterAdjoint parameter_adjoint);

    void launch_non_pressure_forward(void* stream, std::uint32_t particle_count, float support_radius, float gravity_x, float gravity_y, float gravity_z, ConstVector positions, ConstVector velocities, ConstVector external_accelerations, ParticleParameters parameters, Neighborhood neighborhood, Boundary boundary, const float* densities, Vector accelerations);
    void launch_non_pressure_jvp(void* stream, std::uint32_t particle_count, float support_radius, ConstVector positions, ConstVector velocities, ConstVector external_acceleration_tangent, ConstVector position_tangent, ConstVector velocity_tangent, ParticleParameters parameters, ParticleParameterTangent parameter_tangent, Neighborhood neighborhood, Boundary boundary, const float* densities, const float* density_tangent, Vector acceleration_tangent);
    void launch_non_pressure_vjp(void* stream, std::uint32_t particle_count, float support_radius, ConstVector positions, ConstVector velocities, ParticleParameters parameters, Neighborhood neighborhood, Boundary boundary, const float* densities, ConstVectorAdjoint acceleration_adjoint, VectorAdjoint position_adjoint, VectorAdjoint velocity_adjoint, VectorAdjoint control_adjoint, double* density_adjoint, ParticleParameterAdjoint parameter_adjoint);

    void launch_integrate_forward(void* stream, std::uint32_t particle_count, float time_step, Domain domain, ConstVector positions, ConstVector velocities, ConstVector accelerations, Vector next_positions, Vector next_velocities);
    void launch_integrate_jvp(void* stream, std::uint32_t particle_count, float time_step, Domain domain, ConstVector positions, ConstVector velocities, ConstVector accelerations, ConstVector position_tangent, ConstVector velocity_tangent, ConstVector acceleration_tangent, Vector next_position_tangent, Vector next_velocity_tangent);
    void launch_integrate_vjp(void* stream, std::uint32_t particle_count, float time_step, Domain domain, ConstVector positions, ConstVector velocities, ConstVector accelerations, ConstVectorAdjoint next_position_adjoint, ConstVectorAdjoint next_velocity_adjoint, VectorAdjoint position_adjoint, VectorAdjoint velocity_adjoint, VectorAdjoint acceleration_adjoint);

    void launch_pressure_acceleration_forward(void* stream, std::uint32_t particle_count, float support_radius, ConstVector positions, ParticleParameters parameters, Neighborhood neighborhood, Boundary boundary, const float* densities, const float* pressures, Vector accelerations);
    void launch_pressure_acceleration_jvp(void* stream, std::uint32_t particle_count, float support_radius, ConstVector positions, ConstVector position_tangent, ParticleParameters parameters, ParticleParameterTangent parameter_tangent, Neighborhood neighborhood, Boundary boundary, const float* densities, const float* density_tangent, const float* pressures, const float* pressure_tangent, Vector acceleration_tangent);
    void launch_pressure_acceleration_vjp(void* stream, std::uint32_t particle_count, float support_radius, ConstVector positions, ParticleParameters parameters, Neighborhood neighborhood, Boundary boundary, const float* densities, const float* pressures, ConstVectorAdjoint acceleration_adjoint, VectorAdjoint position_adjoint, double* density_adjoint, double* pressure_adjoint, ParticleParameterAdjoint parameter_adjoint);

    void launch_add_acceleration(void* stream, std::uint32_t particle_count, ConstVector first, ConstVector second, Vector output);
    void launch_add_acceleration_adjoint(void* stream, std::uint32_t particle_count, ConstVectorAdjoint output_adjoint, VectorAdjoint first_adjoint, VectorAdjoint second_adjoint);

} // namespace xayah::fluid::sph::cuda_kernel

#endif
