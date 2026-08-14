#ifndef XAYAH_FLUID_GRID_H
#define XAYAH_FLUID_GRID_H

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace xayah::fluid::grid::cuda_kernels {

    struct Configuration {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        float origin_x;
        float origin_y;
        float origin_z;
        float cell_size;
        float time_step;
        std::uint32_t pressure_iterations;
        std::uint32_t extrapolation_iterations;
        std::uint32_t pressure_anchor;
        bool pressure_anchor_enabled;
    };

    struct ConstVectorField {
        const float* x;
        const float* y;
        const float* z;
    };

    struct VectorField {
        float* x;
        float* y;
        float* z;

        operator ConstVectorField() const { return {.x = x, .y = y, .z = z}; }
    };

    struct ConstVectorAdjointField {
        const double* x;
        const double* y;
        const double* z;
    };

    struct VectorAdjointField {
        double* x;
        double* y;
        double* z;

        operator ConstVectorAdjointField() const { return {.x = x, .y = y, .z = z}; }
    };

    struct ConstMatrixField {
        const float* values[9];
    };

    struct MatrixField {
        float* values[9];

        operator ConstMatrixField() const {
            return {.values = {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]}};
        }
    };

    struct ConstMatrixAdjointField {
        const double* values[9];
    };

    struct MatrixAdjointField {
        double* values[9];

        operator ConstMatrixAdjointField() const {
            return {.values = {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]}};
        }
    };

    struct ConstStaggeredField {
        const float* x;
        const float* y;
        const float* z;
    };

    struct StaggeredField {
        float* x;
        float* y;
        float* z;

        operator ConstStaggeredField() const { return {.x = x, .y = y, .z = z}; }
    };

    struct ConstStaggeredAdjointField {
        const double* x;
        const double* y;
        const double* z;
    };

    struct StaggeredAdjointField {
        double* x;
        double* y;
        double* z;

        operator ConstStaggeredAdjointField() const { return {.x = x, .y = y, .z = z}; }
    };

    std::size_t sort_scratch_bytes(std::size_t contribution_count);
    std::size_t reduction_scratch_bytes(std::size_t value_count);
    void sort_contributions(cudaStream_t stream, void* scratch, std::size_t scratch_bytes, const std::uint64_t* keys, std::uint64_t* sorted_keys, const std::uint32_t* ids, std::uint32_t* sorted_ids, std::size_t count);
    void cell_topology(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, ConstVectorField positions, std::uint64_t* keys, std::uint32_t* ids);
    void marker_forward(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, const std::uint64_t* sorted_cell_keys, const std::uint32_t* domain_cell_types, std::uint32_t* cell_types);
    void p2g_topology(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, ConstVectorField positions, std::uint64_t* keys, std::uint32_t* ids);
    void p2g_forward(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, bool apic, ConstVectorField positions, ConstVectorField velocities, ConstMatrixField affine, ConstVectorField accelerations, const float* masses, float gravity_x, float gravity_y, float gravity_z, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, float* face_mass, StaggeredField old_velocity, StaggeredField forced_velocity);
    void p2g_jvp(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, bool apic, ConstVectorField positions, ConstVectorField velocities, ConstMatrixField affine, ConstVectorField accelerations, const float* masses, float gravity_x, float gravity_y, float gravity_z, ConstVectorField position_tangent, ConstVectorField velocity_tangent, ConstMatrixField affine_tangent, ConstVectorField acceleration_tangent, const float* mass_tangent, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const float* face_mass, ConstStaggeredField old_velocity, ConstStaggeredField forced_velocity, StaggeredField old_velocity_tangent, StaggeredField forced_velocity_tangent);
    void p2g_vjp(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, bool apic, ConstVectorField positions, ConstVectorField velocities, ConstMatrixField affine, ConstVectorField accelerations, const float* masses, float gravity_x, float gravity_y, float gravity_z, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const float* face_mass, ConstStaggeredField old_velocity, ConstStaggeredField forced_velocity, ConstStaggeredAdjointField old_velocity_adjoint, ConstStaggeredAdjointField forced_velocity_adjoint, VectorAdjointField position_adjoint, VectorAdjointField velocity_adjoint, MatrixAdjointField affine_adjoint, VectorAdjointField acceleration_adjoint, double* mass_adjoint);
    void projection_forward(cudaStream_t stream, Configuration configuration, const std::uint32_t* cell_types, ConstStaggeredField solid_velocity, ConstStaggeredField velocity, float* divergence, float* pressure_history, StaggeredField projected_velocity);
    void projection_jvp(cudaStream_t stream, Configuration configuration, const std::uint32_t* cell_types, ConstStaggeredField velocity_tangent, float* divergence_tangent, float* pressure_tangent_history, StaggeredField projected_velocity_tangent);
    void projection_vjp(cudaStream_t stream, Configuration configuration, const std::uint32_t* cell_types, ConstStaggeredAdjointField projected_velocity_adjoint, double* divergence_adjoint, double* pressure_adjoint_history, StaggeredAdjointField velocity_adjoint);
    void extrapolation_forward(cudaStream_t stream, Configuration configuration, const std::uint32_t* cell_types, ConstStaggeredField solid_velocity, ConstStaggeredField projected_velocity, std::uint32_t* valid_history, float* velocity_history, StaggeredField extrapolated_velocity);
    void extrapolation_jvp(cudaStream_t stream, Configuration configuration, const std::uint32_t* cell_types, const std::uint32_t* valid_history, ConstStaggeredField projected_velocity_tangent, float* velocity_tangent_history, StaggeredField extrapolated_velocity_tangent);
    void extrapolation_vjp(cudaStream_t stream, Configuration configuration, const std::uint32_t* cell_types, const std::uint32_t* valid_history, ConstStaggeredAdjointField extrapolated_velocity_adjoint, double* velocity_adjoint_history, StaggeredAdjointField projected_velocity_adjoint);
    void g2p_pic_flip_forward(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, ConstVectorField positions, ConstVectorField particle_velocity, ConstStaggeredField old_grid_velocity, ConstStaggeredField new_grid_velocity, const float* blend, VectorField velocity);
    void g2p_pic_flip_jvp(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, ConstVectorField positions, ConstVectorField particle_velocity, ConstStaggeredField old_grid_velocity, ConstStaggeredField new_grid_velocity, const float* blend, ConstVectorField position_tangent, ConstVectorField particle_velocity_tangent, ConstStaggeredField old_grid_velocity_tangent, ConstStaggeredField new_grid_velocity_tangent, const float* blend_tangent, VectorField velocity_tangent);
    void g2p_pic_flip_vjp(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, ConstVectorField positions, ConstVectorField particle_velocity, ConstStaggeredField old_grid_velocity, ConstStaggeredField new_grid_velocity, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, const float* blend, ConstVectorAdjointField velocity_adjoint, VectorAdjointField position_adjoint, VectorAdjointField particle_velocity_adjoint, StaggeredAdjointField old_grid_velocity_adjoint, StaggeredAdjointField new_grid_velocity_adjoint, double* reduction_values, void* reduction_scratch, std::size_t reduction_scratch_bytes, double* reduction_result, double* blend_adjoint);
    void g2p_apic_forward(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, ConstVectorField positions, ConstStaggeredField grid_velocity, VectorField velocity, MatrixField affine);
    void g2p_apic_jvp(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, ConstVectorField positions, ConstStaggeredField grid_velocity, ConstVectorField position_tangent, ConstStaggeredField grid_velocity_tangent, VectorField velocity_tangent, MatrixField affine_tangent);
    void g2p_apic_vjp(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, ConstVectorField positions, ConstStaggeredField grid_velocity, const std::uint64_t* sorted_keys, const std::uint32_t* sorted_ids, ConstVectorAdjointField velocity_adjoint, ConstMatrixAdjointField affine_adjoint, VectorAdjointField position_adjoint, StaggeredAdjointField grid_velocity_adjoint);
    void advect_forward(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, float particle_radius, const std::uint32_t* cell_types, ConstStaggeredField solid_velocity, ConstVectorField positions, ConstVectorField velocities, std::uint32_t* collision_masks, VectorField unconstrained_positions, VectorField next_positions, VectorField next_velocities);
    void advect_jvp(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, float particle_radius, const std::uint32_t* collision_masks, ConstVectorField position_tangent, ConstVectorField velocity_tangent, VectorField next_position_tangent, VectorField next_velocity_tangent);
    void advect_vjp(cudaStream_t stream, Configuration configuration, std::uint32_t particle_count, float particle_radius, const std::uint32_t* collision_masks, ConstVectorAdjointField next_position_adjoint, ConstVectorAdjointField next_velocity_adjoint, VectorAdjointField position_adjoint, VectorAdjointField velocity_adjoint);
    void accumulate_vector_adjoint(cudaStream_t stream, std::size_t count, ConstVectorAdjointField source, VectorAdjointField destination);
    void accumulate_staggered_adjoint(cudaStream_t stream, std::size_t x_count, std::size_t y_count, std::size_t z_count, ConstStaggeredAdjointField source, StaggeredAdjointField destination);
    void accumulate_matrix_adjoint(cudaStream_t stream, std::size_t count, const double* const source[9], double* const destination[9]);

} // namespace xayah::fluid::grid::cuda_kernels

#endif
