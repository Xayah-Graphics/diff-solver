module;

#include "kernels.cuh"

module xayah.cloth.operators;

import xayah.cloth.data;
import xayah.cloth.runtime;

namespace xayah::cloth {

    namespace {

        cuda_kernel::ConstField kernel_field(const VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernel::Field kernel_field(VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernel::SpringTopology kernel_topology(const DeviceSpringTopology& topology) {
            return {
                .first        = topology.first.data,
                .second       = topology.second.data,
                .offsets      = topology.offsets.data,
                .indices      = topology.indices.data,
                .others       = topology.others.data,
                .signs        = topology.signs.data,
                .spring_count = static_cast<std::uint32_t>(topology.first.size),
            };
        }

    } // namespace

    void ForceAssemblyOperator::forward(Resource& resource, const DeviceTopology& topology, const Configuration& configuration, const State& state, const Control& control, const Parameters& parameters, Forces& forces) const {
        cuda_kernel::launch_force_forward(resource.native_stream, static_cast<std::uint32_t>(parameters.masses.size), configuration.gravity.x, configuration.gravity.y, configuration.gravity.z, kernel_field(state.positions), kernel_field(state.velocities), kernel_field(control.external_forces), parameters.masses.data, kernel_topology(topology.stretch), {.stiffnesses = parameters.stretch_stiffnesses.data, .dampings = parameters.stretch_dampings.data, .rest_lengths = parameters.stretch_rest_lengths.data}, kernel_topology(topology.bending), {.stiffnesses = parameters.bending_stiffnesses.data, .dampings = parameters.bending_dampings.data, .rest_lengths = parameters.bending_rest_lengths.data}, kernel_field(forces.values));
    }

    void ForceAssemblyOperator::jvp(Resource& resource, const DeviceTopology& topology, const Configuration& configuration, const State& state, const Parameters& parameters, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, ForceTangent& force_tangent) const {
        cuda_kernel::launch_force_jvp(resource.native_stream, static_cast<std::uint32_t>(parameters.masses.size), configuration.gravity.x, configuration.gravity.y, configuration.gravity.z, kernel_field(state.positions), kernel_field(state.velocities), kernel_field(control_tangent.external_forces), kernel_field(state_tangent.positions), kernel_field(state_tangent.velocities), parameter_tangent.masses.data, kernel_topology(topology.stretch), {.stiffnesses = parameters.stretch_stiffnesses.data, .dampings = parameters.stretch_dampings.data, .rest_lengths = parameters.stretch_rest_lengths.data}, {.stiffnesses = parameter_tangent.stretch_stiffnesses.data, .dampings = parameter_tangent.stretch_dampings.data, .rest_lengths = parameter_tangent.stretch_rest_lengths.data}, kernel_topology(topology.bending), {.stiffnesses = parameters.bending_stiffnesses.data, .dampings = parameters.bending_dampings.data, .rest_lengths = parameters.bending_rest_lengths.data},
            {.stiffnesses = parameter_tangent.bending_stiffnesses.data, .dampings = parameter_tangent.bending_dampings.data, .rest_lengths = parameter_tangent.bending_rest_lengths.data}, kernel_field(force_tangent.values));
    }

    void ForceAssemblyOperator::vjp(Resource& resource, const DeviceTopology& topology, const Configuration& configuration, const State& state, const Parameters& parameters, const ForceAdjoint& force_adjoint, StateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint) const {
        cuda_kernel::launch_force_state_vjp(resource.native_stream, static_cast<std::uint32_t>(parameters.masses.size), configuration.gravity.x, configuration.gravity.y, configuration.gravity.z, kernel_field(state.positions), kernel_field(state.velocities), kernel_field(force_adjoint.values), kernel_topology(topology.stretch), {.stiffnesses = parameters.stretch_stiffnesses.data, .dampings = parameters.stretch_dampings.data, .rest_lengths = parameters.stretch_rest_lengths.data}, kernel_topology(topology.bending), {.stiffnesses = parameters.bending_stiffnesses.data, .dampings = parameters.bending_dampings.data, .rest_lengths = parameters.bending_rest_lengths.data}, kernel_field(state_adjoint.positions), kernel_field(state_adjoint.velocities), kernel_field(control_adjoint.external_forces), parameter_adjoint.masses.data);

        cuda_kernel::launch_force_parameter_vjp(resource.native_stream, static_cast<std::uint32_t>(topology.stretch.first.size), kernel_field(state.positions), kernel_field(state.velocities), kernel_field(force_adjoint.values), kernel_topology(topology.stretch), {.stiffnesses = parameters.stretch_stiffnesses.data, .dampings = parameters.stretch_dampings.data, .rest_lengths = parameters.stretch_rest_lengths.data}, {.stiffnesses = parameter_adjoint.stretch_stiffnesses.data, .dampings = parameter_adjoint.stretch_dampings.data, .rest_lengths = parameter_adjoint.stretch_rest_lengths.data});

        cuda_kernel::launch_force_parameter_vjp(resource.native_stream, static_cast<std::uint32_t>(topology.bending.first.size), kernel_field(state.positions), kernel_field(state.velocities), kernel_field(force_adjoint.values), kernel_topology(topology.bending), {.stiffnesses = parameters.bending_stiffnesses.data, .dampings = parameters.bending_dampings.data, .rest_lengths = parameters.bending_rest_lengths.data}, {.stiffnesses = parameter_adjoint.bending_stiffnesses.data, .dampings = parameter_adjoint.bending_dampings.data, .rest_lengths = parameter_adjoint.bending_rest_lengths.data});
    }

    void SemiImplicitEulerOperator::forward(Resource& resource, const Configuration& configuration, const State& state, const Parameters& parameters, const Forces& forces, State& integrated_state) const {
        cuda_kernel::launch_euler_forward(resource.native_stream, static_cast<std::uint32_t>(parameters.masses.size), configuration.time_step, kernel_field(state.positions), kernel_field(state.velocities), kernel_field(forces.values), parameters.masses.data, kernel_field(integrated_state.positions), kernel_field(integrated_state.velocities));
    }

    void SemiImplicitEulerOperator::jvp(Resource& resource, const Configuration& configuration, const Parameters& parameters, const Forces& forces, const StateTangent& state_tangent, const ParameterTangent& parameter_tangent, const ForceTangent& force_tangent, StateTangent& integrated_state_tangent) const {
        cuda_kernel::launch_euler_jvp(resource.native_stream, static_cast<std::uint32_t>(parameters.masses.size), configuration.time_step, kernel_field(forces.values), parameters.masses.data, kernel_field(state_tangent.positions), kernel_field(state_tangent.velocities), kernel_field(force_tangent.values), parameter_tangent.masses.data, kernel_field(integrated_state_tangent.positions), kernel_field(integrated_state_tangent.velocities));
    }

    void SemiImplicitEulerOperator::vjp(Resource& resource, const Configuration& configuration, const Parameters& parameters, const Forces& forces, const StateAdjoint& integrated_state_adjoint, StateAdjoint& state_adjoint, ForceAdjoint& force_adjoint, ParameterAdjoint& parameter_adjoint) const {
        cuda_kernel::launch_euler_vjp(resource.native_stream, static_cast<std::uint32_t>(parameters.masses.size), configuration.time_step, kernel_field(forces.values), parameters.masses.data, kernel_field(integrated_state_adjoint.positions), kernel_field(integrated_state_adjoint.velocities), kernel_field(state_adjoint.positions), kernel_field(state_adjoint.velocities), kernel_field(force_adjoint.values), parameter_adjoint.masses.data);
    }

    void FixedConstraintOperator::forward(Resource& resource, const DeviceTopology& topology, const State& state, State& constrained_state) const {
        cuda_kernel::launch_constraint_forward(resource.native_stream, static_cast<std::uint32_t>(topology.anchor_mask.size), topology.anchor_mask.data, kernel_field(topology.anchor_positions), kernel_field(state.positions), kernel_field(state.velocities), kernel_field(constrained_state.positions), kernel_field(constrained_state.velocities));
    }

    void FixedConstraintOperator::jvp(Resource& resource, const DeviceTopology& topology, const StateTangent& state_tangent, StateTangent& constrained_state_tangent) const {
        cuda_kernel::launch_constraint_jvp(resource.native_stream, static_cast<std::uint32_t>(topology.anchor_mask.size), topology.anchor_mask.data, kernel_field(state_tangent.positions), kernel_field(state_tangent.velocities), kernel_field(constrained_state_tangent.positions), kernel_field(constrained_state_tangent.velocities));
    }

    void FixedConstraintOperator::vjp(Resource& resource, const DeviceTopology& topology, const StateAdjoint& constrained_state_adjoint, StateAdjoint& state_adjoint) const {
        cuda_kernel::launch_constraint_vjp(resource.native_stream, static_cast<std::uint32_t>(topology.anchor_mask.size), topology.anchor_mask.data, kernel_field(constrained_state_adjoint.positions), kernel_field(constrained_state_adjoint.velocities), kernel_field(state_adjoint.positions), kernel_field(state_adjoint.velocities));
    }

} // namespace xayah::cloth
