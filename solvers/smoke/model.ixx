export module xayah.smoke.model;

import std;
import xayah.cuda;
import xayah.smoke.data;
import xayah.smoke.operators;

export namespace xayah::smoke {

    enum class ExecutionMode : std::uint32_t {
        forward,
        differentiable,
    };

    struct Model;

    struct ExecutionContext {
    public:
        std::shared_ptr<cuda::Resource> resource;
        DeviceDomain domain;

    private:
        std::size_t cell_count_;
        std::array<std::size_t, 3u> face_counts_;
        StaggeredVectorField raw_advected_velocity_;
        ScalarField sourced_density_tangent_;
        ScalarField sourced_temperature_tangent_;
        CenteredVectorField force_tangent_;
        VorticityCache vorticity_tangent_scratch_;
        VorticityAdjointCache vorticity_adjoint_scratch_;
        StaggeredVectorField forced_velocity_tangent_;
        StaggeredVectorField raw_advected_velocity_tangent_;
        StaggeredVectorField advected_velocity_tangent_;
        ScalarField pressure_;
        ScalarField pressure_rhs_;
        ScalarField pressure_tangent_;
        ScalarField pressure_rhs_tangent_;
        ScalarAdjointField sourced_density_adjoint_;
        ScalarAdjointField sourced_temperature_adjoint_;
        CenteredVectorAdjointField force_adjoint_;
        StaggeredVectorAdjointField projected_velocity_adjoint_;
        StaggeredVectorAdjointField advected_velocity_adjoint_;
        StaggeredVectorAdjointField raw_advected_velocity_adjoint_;
        StaggeredVectorAdjointField forced_velocity_adjoint_;
        ScalarAdjointField pressure_adjoint_;
        ScalarAdjointField pressure_rhs_adjoint_;

        friend struct Model;
    };

    struct Model {
    public:
        const Configuration configuration;

        explicit Model(Configuration configuration);

        [[nodiscard]] ExecutionContext allocate_context(ExecutionMode mode) const;

    private:
        [[nodiscard]] ScalarField allocate_scalar_field(ExecutionContext& context) const;
        [[nodiscard]] CenteredVectorField allocate_centered_vector_field(ExecutionContext& context) const;
        [[nodiscard]] StaggeredVectorField allocate_staggered_vector_field(ExecutionContext& context) const;
        [[nodiscard]] VorticityCache allocate_vorticity_cache(ExecutionContext& context) const;
        [[nodiscard]] ScalarAdjointField allocate_scalar_adjoint_field(ExecutionContext& context) const;
        [[nodiscard]] CenteredVectorAdjointField allocate_centered_vector_adjoint_field(ExecutionContext& context) const;
        [[nodiscard]] StaggeredVectorAdjointField allocate_staggered_vector_adjoint_field(ExecutionContext& context) const;
        [[nodiscard]] VorticityAdjointCache allocate_vorticity_adjoint_cache(ExecutionContext& context) const;

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
        SourceOperator source_;
        ForceAssemblyOperator force_;
        VelocityEvolutionOperator velocity_;
        PressureProjectionOperator projection_;
        ScalarAdvectionOperator scalar_advection_;
    };

} // namespace xayah::smoke
