export module xayah.cloth.model;

import std;
import xayah.cloth.data;
import xayah.cloth.operators;
import xayah.cloth.runtime;

export namespace xayah::cloth {

    struct Model;

    struct ExecutionContext {
    private:
        std::shared_ptr<Resource> resource_owner_;
        ExecutionContext(const Configuration& configuration, const Topology& topology);

    public:
        Resource& resource;
        DeviceTopology device_topology;

        void upload(std::span<const float> source, Buffer<float>& destination);
        void download(const Buffer<float>& source, std::span<float> destination);
        void upload(std::span<const Vector3> source, VectorField& destination);
        void download(const VectorField& source, std::span<Vector3> destination);
        void synchronize();

    private:
        [[nodiscard]] Buffer<float> make_scalar_buffer(std::size_t size) const;
        [[nodiscard]] Buffer<std::uint32_t> make_index_buffer(std::size_t size) const;
        [[nodiscard]] VectorField make_vector_field(std::size_t size) const;
        void zero(Buffer<float>& buffer);
        void zero(VectorField& field);
        void copy(const Buffer<float>& source, Buffer<float>& destination);
        void copy(const VectorField& source, VectorField& destination);
        void accumulate(const VectorField& source, VectorField& destination);

        State integrated_state_;
        ForceTangent force_tangent_;
        StateTangent integrated_state_tangent_;
        ForceAdjoint force_adjoint_;
        StateAdjoint integrated_state_adjoint_;

        friend struct Model;
    };

    struct Model {
        const Configuration configuration;
        const Topology topology;

        explicit Model(Configuration configuration);

        [[nodiscard]] ExecutionContext make_context() const;

        [[nodiscard]] State make_state(ExecutionContext& context) const;
        [[nodiscard]] Control make_control(ExecutionContext& context) const;
        [[nodiscard]] Parameters make_parameters(ExecutionContext& context) const;
        [[nodiscard]] StepCache make_step_cache(ExecutionContext& context) const;
        [[nodiscard]] StateTangent make_state_tangent(ExecutionContext& context) const;
        [[nodiscard]] ControlTangent make_control_tangent(ExecutionContext& context) const;
        [[nodiscard]] ParameterTangent make_parameter_tangent(ExecutionContext& context) const;
        [[nodiscard]] StateAdjoint make_state_adjoint(ExecutionContext& context) const;
        [[nodiscard]] ControlAdjoint make_control_adjoint(ExecutionContext& context) const;
        [[nodiscard]] ParameterAdjoint make_parameter_adjoint(ExecutionContext& context) const;

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
