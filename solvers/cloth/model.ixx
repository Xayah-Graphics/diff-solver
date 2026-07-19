export module xayah.cloth.model;

import std;
import xayah.cloth.data;
import xayah.cloth.operators;
import xayah.cuda;

export namespace xayah::cloth {

    enum class ExecutionMode : std::uint32_t {
        forward,
        differentiable,
    };

    struct Model;

    struct ExecutionContext {
    public:
        std::shared_ptr<cuda::Resource> resource;
        DeviceTopology device_topology;

    private:
        State integrated_state_;
        ForceTangent force_tangent_;
        StateTangent integrated_state_tangent_;
        ForceAdjoint force_adjoint_;
        StateAdjoint integrated_state_adjoint_;

        friend struct Model;
    };

    void upload(cuda::Resource& resource, std::span<const float> source, cuda::Buffer<float>& destination);
    void upload(cuda::Resource& resource, std::span<const Vector3> source, VectorField& destination);
    void download(cuda::Resource& resource, const VectorField& source, std::span<Vector3> destination);

    struct Model {
    public:
        const Configuration configuration;
        const Topology topology;

        explicit Model(Configuration configuration);

        [[nodiscard]] ExecutionContext allocate_context(ExecutionMode mode) const;

    private:
        [[nodiscard]] VectorField allocate_vector_field(ExecutionContext& context, std::size_t size) const;

    public:
        [[nodiscard]] State allocate_state(ExecutionContext& context) const;
        [[nodiscard]] Control allocate_control(ExecutionContext& context) const;
        [[nodiscard]] Parameters allocate_parameters(ExecutionContext& context) const;
        [[nodiscard]] StepCache allocate_step_cache(ExecutionContext& context) const;
        [[nodiscard]] StateTangent allocate_state_tangent(ExecutionContext& context) const;
        [[nodiscard]] ControlTangent allocate_control_tangent(ExecutionContext& context) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(ExecutionContext& context) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(ExecutionContext& context) const;
        [[nodiscard]] ControlAdjoint allocate_control_adjoint(ExecutionContext& context) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(ExecutionContext& context) const;

        void copy_state(const State& source, State& destination, ExecutionContext& context) const;
        void copy_state_tangent(const StateTangent& source, StateTangent& destination, ExecutionContext& context) const;
        void copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const;
        void accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const;

        void forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& step_cache, ExecutionContext& context) const;
        void jvp_step(const State& state, const Control& control, const Parameters& parameters, const State& next_state, const StepCache& step_cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const;
        void vjp_step(const State& state, const Control& control, const Parameters& parameters, const State& next_state, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const;

    private:
        ForceAssemblyOperator force_assembly_;
        SemiImplicitEulerOperator semi_implicit_euler_;
        FixedConstraintOperator fixed_constraint_;
    };

} // namespace xayah::cloth
