module xayah.spring_chain.operators;

import std;
import xayah.spring_chain.data;

namespace xayah::spring_chain {

    template <Scalar T>
    void ForceAssemblyOperator<T>::forward(const Configuration<T>& configuration, const State<T>& state, const Control<T>& control, const Parameters<T>& parameters, Forces<T>& forces) const {
        forces.values = control.external_forces;
        for (std::size_t edge_index = 0; edge_index < configuration.edges.size(); ++edge_index) {
            const Edge& edge          = configuration.edges[edge_index];
            const T extension         = state.positions[edge.second] - state.positions[edge.first] - parameters.rest_lengths[edge_index];
            const T relative_velocity = state.velocities[edge.second] - state.velocities[edge.first];
            const T edge_force        = parameters.stiffnesses[edge_index] * extension + parameters.dampings[edge_index] * relative_velocity;
            forces.values[edge.first] += edge_force;
            forces.values[edge.second] -= edge_force;
        }
    }

    template <Scalar T>
    void ForceAssemblyOperator<T>::jvp(const Configuration<T>& configuration, const State<T>& state, const Parameters<T>& parameters, const StateTangent<T>& state_tangent, const ControlTangent<T>& control_tangent, const ParameterTangent<T>& parameter_tangent, ForceTangent<T>& force_tangent) const {
        force_tangent.values = control_tangent.external_forces;
        for (std::size_t edge_index = 0; edge_index < configuration.edges.size(); ++edge_index) {
            const Edge& edge                  = configuration.edges[edge_index];
            const T extension                 = state.positions[edge.second] - state.positions[edge.first] - parameters.rest_lengths[edge_index];
            const T extension_tangent         = state_tangent.positions[edge.second] - state_tangent.positions[edge.first] - parameter_tangent.rest_lengths[edge_index];
            const T relative_velocity         = state.velocities[edge.second] - state.velocities[edge.first];
            const T relative_velocity_tangent = state_tangent.velocities[edge.second] - state_tangent.velocities[edge.first];
            const T edge_force_tangent        = parameter_tangent.stiffnesses[edge_index] * extension + parameters.stiffnesses[edge_index] * extension_tangent + parameter_tangent.dampings[edge_index] * relative_velocity + parameters.dampings[edge_index] * relative_velocity_tangent;
            force_tangent.values[edge.first] += edge_force_tangent;
            force_tangent.values[edge.second] -= edge_force_tangent;
        }
    }

    template <Scalar T>
    void ForceAssemblyOperator<T>::vjp(const Configuration<T>& configuration, const State<T>& state, const Parameters<T>& parameters, const ForceAdjoint<T>& force_adjoint, StateAdjoint<T>& state_adjoint, ControlAdjoint<T>& control_adjoint, ParameterAdjoint<T>& parameter_adjoint) const {
        for (std::size_t particle = 0; particle < force_adjoint.values.size(); ++particle) control_adjoint.external_forces[particle] += force_adjoint.values[particle];
        for (std::size_t edge_index = 0; edge_index < configuration.edges.size(); ++edge_index) {
            const Edge& edge           = configuration.edges[edge_index];
            const T edge_force_adjoint = force_adjoint.values[edge.first] - force_adjoint.values[edge.second];
            const T extension          = state.positions[edge.second] - state.positions[edge.first] - parameters.rest_lengths[edge_index];
            const T relative_velocity  = state.velocities[edge.second] - state.velocities[edge.first];
            state_adjoint.positions[edge.first] -= parameters.stiffnesses[edge_index] * edge_force_adjoint;
            state_adjoint.positions[edge.second] += parameters.stiffnesses[edge_index] * edge_force_adjoint;
            state_adjoint.velocities[edge.first] -= parameters.dampings[edge_index] * edge_force_adjoint;
            state_adjoint.velocities[edge.second] += parameters.dampings[edge_index] * edge_force_adjoint;
            parameter_adjoint.stiffnesses[edge_index] += extension * edge_force_adjoint;
            parameter_adjoint.dampings[edge_index] += relative_velocity * edge_force_adjoint;
            parameter_adjoint.rest_lengths[edge_index] -= parameters.stiffnesses[edge_index] * edge_force_adjoint;
        }
    }

    template class ForceAssemblyOperator<float>;
    template class ForceAssemblyOperator<double>;

    template <Scalar T>
    void SemiImplicitEulerOperator<T>::forward(const Configuration<T>& configuration, const State<T>& state, const Parameters<T>& parameters, const Forces<T>& forces, State<T>& integrated_state) const {
        for (std::size_t particle = 0; particle < state.positions.size(); ++particle) {
            integrated_state.velocities[particle] = state.velocities[particle] + configuration.time_step * forces.values[particle] / parameters.masses[particle];
            integrated_state.positions[particle]  = state.positions[particle] + configuration.time_step * integrated_state.velocities[particle];
        }
    }

    template <Scalar T>
    void SemiImplicitEulerOperator<T>::jvp(const Configuration<T>& configuration, const Parameters<T>& parameters, const Forces<T>& forces, const StateTangent<T>& state_tangent, const ParameterTangent<T>& parameter_tangent, const ForceTangent<T>& force_tangent, StateTangent<T>& integrated_state_tangent) const {
        for (std::size_t particle = 0; particle < state_tangent.positions.size(); ++particle) {
            integrated_state_tangent.velocities[particle] = state_tangent.velocities[particle] + configuration.time_step * (force_tangent.values[particle] / parameters.masses[particle] - forces.values[particle] * parameter_tangent.masses[particle] / (parameters.masses[particle] * parameters.masses[particle]));
            integrated_state_tangent.positions[particle]  = state_tangent.positions[particle] + configuration.time_step * integrated_state_tangent.velocities[particle];
        }
    }

    template <Scalar T>
    void SemiImplicitEulerOperator<T>::vjp(const Configuration<T>& configuration, const Parameters<T>& parameters, const Forces<T>& forces, const StateAdjoint<T>& integrated_state_adjoint, StateAdjoint<T>& state_adjoint, ForceAdjoint<T>& force_adjoint, ParameterAdjoint<T>& parameter_adjoint) const {
        for (std::size_t particle = 0; particle < integrated_state_adjoint.positions.size(); ++particle) {
            const T velocity_adjoint = integrated_state_adjoint.velocities[particle] + configuration.time_step * integrated_state_adjoint.positions[particle];
            state_adjoint.positions[particle] += integrated_state_adjoint.positions[particle];
            state_adjoint.velocities[particle] += velocity_adjoint;
            force_adjoint.values[particle] += configuration.time_step * velocity_adjoint / parameters.masses[particle];
            parameter_adjoint.masses[particle] -= configuration.time_step * forces.values[particle] * velocity_adjoint / (parameters.masses[particle] * parameters.masses[particle]);
        }
    }

    template class SemiImplicitEulerOperator<float>;
    template class SemiImplicitEulerOperator<double>;

    template <Scalar T>
    void FixedConstraintOperator<T>::forward(const Configuration<T>& configuration, const State<T>& state, State<T>& constrained_state) const {
        for (std::size_t particle = 0; particle < configuration.anchors.size(); ++particle) {
            if (configuration.anchors[particle].has_value()) {
                constrained_state.positions[particle]  = *configuration.anchors[particle];
                constrained_state.velocities[particle] = T{0};
            } else {
                constrained_state.positions[particle]  = state.positions[particle];
                constrained_state.velocities[particle] = state.velocities[particle];
            }
        }
    }

    template <Scalar T>
    void FixedConstraintOperator<T>::jvp(const Configuration<T>& configuration, const StateTangent<T>& state_tangent, StateTangent<T>& constrained_state_tangent) const {
        for (std::size_t particle = 0; particle < configuration.anchors.size(); ++particle) {
            if (configuration.anchors[particle].has_value()) {
                constrained_state_tangent.positions[particle]  = T{0};
                constrained_state_tangent.velocities[particle] = T{0};
            } else {
                constrained_state_tangent.positions[particle]  = state_tangent.positions[particle];
                constrained_state_tangent.velocities[particle] = state_tangent.velocities[particle];
            }
        }
    }

    template <Scalar T>
    void FixedConstraintOperator<T>::vjp(const Configuration<T>& configuration, const StateAdjoint<T>& constrained_state_adjoint, StateAdjoint<T>& state_adjoint) const {
        for (std::size_t particle = 0; particle < configuration.anchors.size(); ++particle) {
            if (configuration.anchors[particle].has_value()) continue;
            state_adjoint.positions[particle] += constrained_state_adjoint.positions[particle];
            state_adjoint.velocities[particle] += constrained_state_adjoint.velocities[particle];
        }
    }

    template class FixedConstraintOperator<float>;
    template class FixedConstraintOperator<double>;

} // namespace xayah::spring_chain
