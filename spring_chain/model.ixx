export module xayah.spring_chain.model;

import xayah.spring_chain.data;
import xayah.spring_chain.operators;

export namespace xayah::spring_chain {

    template <Scalar T>
    class Model;

    template <Scalar T>
    struct ExecutionContext {
    private:
        explicit ExecutionContext(std::size_t particle_count);

        State<T> integrated_state;
        ForceTangent<T> force_tangent;
        StateTangent<T> integrated_state_tangent;
        ForceAdjoint<T> force_adjoint;
        StateAdjoint<T> integrated_state_adjoint;

        friend class Model<T>;
    };

    extern template struct ExecutionContext<float>;
    extern template struct ExecutionContext<double>;

    template <Scalar T>
    class Model {
    public:
        explicit Model(Configuration<T> configuration);

        [[nodiscard]] const Configuration<T>& configuration() const;
        [[nodiscard]] ExecutionContext<T> make_context() const;
        [[nodiscard]] State<T> make_state() const;
        [[nodiscard]] StepCache<T> make_step_cache() const;
        [[nodiscard]] StateTangent<T> make_state_tangent() const;
        [[nodiscard]] ControlTangent<T> make_control_tangent() const;
        [[nodiscard]] ParameterTangent<T> make_parameter_tangent() const;
        [[nodiscard]] StateAdjoint<T> make_state_adjoint() const;
        [[nodiscard]] ControlAdjoint<T> make_control_adjoint() const;
        [[nodiscard]] ParameterAdjoint<T> make_parameter_adjoint() const;

        void forward_step(const State<T>& state, const Control<T>& control, const Parameters<T>& parameters, State<T>& next_state, StepCache<T>& step_cache, ExecutionContext<T>& context) const;

        void jvp_step(const State<T>& state, const Control<T>& control, const Parameters<T>& parameters, const State<T>& next_state, const StepCache<T>& step_cache, const StateTangent<T>& state_tangent, const ControlTangent<T>& control_tangent, const ParameterTangent<T>& parameter_tangent, StateTangent<T>& next_state_tangent, ExecutionContext<T>& context) const;

        void vjp_step(const State<T>& state, const Control<T>& control, const Parameters<T>& parameters, const State<T>& next_state, const StepCache<T>& step_cache, const StateAdjoint<T>& next_state_adjoint, StateAdjoint<T>& previous_state_adjoint, ControlAdjoint<T>& control_adjoint, ParameterAdjoint<T>& parameter_adjoint, ExecutionContext<T>& context) const;

    private:
        Configuration<T> configuration_;
        ForceAssemblyOperator<T> force_assembly_;
        SemiImplicitEulerOperator<T> semi_implicit_euler_;
        FixedConstraintOperator<T> fixed_constraint_;
    };

    extern template class Model<float>;
    extern template class Model<double>;

} // namespace xayah::spring_chain
