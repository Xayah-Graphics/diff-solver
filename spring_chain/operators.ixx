export module xayah.spring_chain.operators;

import xayah.spring_chain.data;

export namespace xayah::spring_chain {

    template <Scalar T>
    class ForceAssemblyOperator {
    public:
        void forward(const Configuration<T>& configuration, const State<T>& state, const Control<T>& control, const Parameters<T>& parameters, Forces<T>& forces) const;

        void jvp(const Configuration<T>& configuration, const State<T>& state, const Parameters<T>& parameters, const StateTangent<T>& state_tangent, const ControlTangent<T>& control_tangent, const ParameterTangent<T>& parameter_tangent, ForceTangent<T>& force_tangent) const;

        void vjp(const Configuration<T>& configuration, const State<T>& state, const Parameters<T>& parameters, const ForceAdjoint<T>& force_adjoint, StateAdjoint<T>& state_adjoint, ControlAdjoint<T>& control_adjoint, ParameterAdjoint<T>& parameter_adjoint) const;
    };

    extern template class ForceAssemblyOperator<float>;
    extern template class ForceAssemblyOperator<double>;

    template <Scalar T>
    class SemiImplicitEulerOperator {
    public:
        void forward(const Configuration<T>& configuration, const State<T>& state, const Parameters<T>& parameters, const Forces<T>& forces, State<T>& integrated_state) const;

        void jvp(const Configuration<T>& configuration, const Parameters<T>& parameters, const Forces<T>& forces, const StateTangent<T>& state_tangent, const ParameterTangent<T>& parameter_tangent, const ForceTangent<T>& force_tangent, StateTangent<T>& integrated_state_tangent) const;

        void vjp(const Configuration<T>& configuration, const Parameters<T>& parameters, const Forces<T>& forces, const StateAdjoint<T>& integrated_state_adjoint, StateAdjoint<T>& state_adjoint, ForceAdjoint<T>& force_adjoint, ParameterAdjoint<T>& parameter_adjoint) const;
    };

    extern template class SemiImplicitEulerOperator<float>;
    extern template class SemiImplicitEulerOperator<double>;

    template <Scalar T>
    class FixedConstraintOperator {
    public:
        void forward(const Configuration<T>& configuration, const State<T>& state, State<T>& constrained_state) const;

        void jvp(const Configuration<T>& configuration, const StateTangent<T>& state_tangent, StateTangent<T>& constrained_state_tangent) const;

        void vjp(const Configuration<T>& configuration, const StateAdjoint<T>& constrained_state_adjoint, StateAdjoint<T>& state_adjoint) const;
    };

    extern template class FixedConstraintOperator<float>;
    extern template class FixedConstraintOperator<double>;

} // namespace xayah::spring_chain
