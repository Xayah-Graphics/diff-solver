export module xayah.cloth.operators;

import xayah.cloth.data;
import xayah.cuda;

export namespace xayah::cloth {

    struct ForceAssemblyOperator {
        void forward(cuda::Resource& resource, const DeviceTopology& topology, const Configuration& configuration, const State& state, const Control& control, const Parameters& parameters, Forces& forces) const;
        void jvp(cuda::Resource& resource, const DeviceTopology& topology, const Configuration& configuration, const State& state, const Parameters& parameters, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, ForceTangent& force_tangent) const;
        void vjp(cuda::Resource& resource, const DeviceTopology& topology, const Configuration& configuration, const State& state, const Parameters& parameters, const ForceAdjoint& force_adjoint, StateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint) const;
    };

    struct SemiImplicitEulerOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const State& state, const Parameters& parameters, const Forces& forces, State& integrated_state) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, const Parameters& parameters, const Forces& forces, const StateTangent& state_tangent, const ParameterTangent& parameter_tangent, const ForceTangent& force_tangent, StateTangent& integrated_state_tangent) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, const Parameters& parameters, const Forces& forces, const StateAdjoint& integrated_state_adjoint, StateAdjoint& state_adjoint, ForceAdjoint& force_adjoint, ParameterAdjoint& parameter_adjoint) const;
    };

    struct FixedConstraintOperator {
        void forward(cuda::Resource& resource, const DeviceTopology& topology, const State& state, State& constrained_state) const;
        void jvp(cuda::Resource& resource, const DeviceTopology& topology, const StateTangent& state_tangent, StateTangent& constrained_state_tangent) const;
        void vjp(cuda::Resource& resource, const DeviceTopology& topology, const StateAdjoint& constrained_state_adjoint, StateAdjoint& state_adjoint) const;
    };

} // namespace xayah::cloth
