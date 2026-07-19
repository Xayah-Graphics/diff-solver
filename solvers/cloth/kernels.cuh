#ifndef XAYAH_CLOTH_KERNELS_CUH
#define XAYAH_CLOTH_KERNELS_CUH

#include <cstdint>

namespace xayah::cloth::cuda_kernel {

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

    struct SpringTopology {
        const std::uint32_t* first;
        const std::uint32_t* second;
        const std::uint32_t* offsets;
        const std::uint32_t* indices;
        const std::uint32_t* others;
        const float* signs;
        std::uint32_t spring_count;
    };

    struct SpringParameters {
        const float* stiffnesses;
        const float* dampings;
        const float* rest_lengths;
    };

    struct SpringParameterTangents {
        const float* stiffnesses;
        const float* dampings;
        const float* rest_lengths;
    };

    struct SpringParameterAdjoints {
        float* stiffnesses;
        float* dampings;
        float* rest_lengths;
    };

    void launch_force_forward(void* stream, std::uint32_t particle_count, float gravity_x, float gravity_y, float gravity_z, ConstField positions, ConstField velocities, ConstField controls, const float* masses, SpringTopology stretch_topology, SpringParameters stretch_parameters, SpringTopology bending_topology, SpringParameters bending_parameters, Field forces);

    void launch_force_jvp(void* stream, std::uint32_t particle_count, float gravity_x, float gravity_y, float gravity_z, ConstField positions, ConstField velocities, ConstField controls_tangent, ConstField positions_tangent, ConstField velocities_tangent, const float* masses_tangent, SpringTopology stretch_topology, SpringParameters stretch_parameters, SpringParameterTangents stretch_tangents, SpringTopology bending_topology, SpringParameters bending_parameters, SpringParameterTangents bending_tangents, Field force_tangent);

    void launch_force_state_vjp(void* stream, std::uint32_t particle_count, float gravity_x, float gravity_y, float gravity_z, ConstField positions, ConstField velocities, ConstField force_adjoint, SpringTopology stretch_topology, SpringParameters stretch_parameters, SpringTopology bending_topology, SpringParameters bending_parameters, Field state_position_adjoint, Field state_velocity_adjoint, Field control_adjoint, float* mass_adjoint);

    void launch_force_parameter_vjp(void* stream, std::uint32_t spring_count, ConstField positions, ConstField velocities, ConstField force_adjoint, SpringTopology topology, SpringParameters parameters, SpringParameterAdjoints parameter_adjoint);

    void launch_euler_forward(void* stream, std::uint32_t particle_count, float time_step, ConstField positions, ConstField velocities, ConstField forces, const float* masses, Field integrated_positions, Field integrated_velocities);

    void launch_euler_jvp(void* stream, std::uint32_t particle_count, float time_step, ConstField forces, const float* masses, ConstField position_tangent, ConstField velocity_tangent, ConstField force_tangent, const float* mass_tangent, Field integrated_position_tangent, Field integrated_velocity_tangent);

    void launch_euler_vjp(void* stream, std::uint32_t particle_count, float time_step, ConstField forces, const float* masses, ConstField integrated_position_adjoint, ConstField integrated_velocity_adjoint, Field state_position_adjoint, Field state_velocity_adjoint, Field force_adjoint, float* mass_adjoint);

    void launch_constraint_forward(void* stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, ConstField anchor_positions, ConstField positions, ConstField velocities, Field constrained_positions, Field constrained_velocities);

    void launch_constraint_jvp(void* stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, ConstField position_tangent, ConstField velocity_tangent, Field constrained_position_tangent, Field constrained_velocity_tangent);

    void launch_constraint_vjp(void* stream, std::uint32_t particle_count, const std::uint32_t* anchor_mask, ConstField constrained_position_adjoint, ConstField constrained_velocity_adjoint, Field position_adjoint, Field velocity_adjoint);

    void launch_accumulate(void* stream, std::uint32_t count, ConstField source, Field destination);

} // namespace xayah::cloth::cuda_kernel

#endif
