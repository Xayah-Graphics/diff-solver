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
    private:
        std::shared_ptr<cuda::Resource> resource_owner_;
        ExecutionContext(const Configuration& configuration, ExecutionMode mode);

    public:
        cuda::Resource& resource;
        DeviceDomain domain;

        void upload(std::span<const float> source, cuda::Buffer<float>& destination);
        void download(const cuda::Buffer<float>& source, std::span<float> destination);
        void upload(std::span<const double> source, cuda::Buffer<double>& destination);
        void download(const cuda::Buffer<double>& source, std::span<double> destination);
        void upload(float source, cuda::Buffer<float>& destination);
        void synchronize();

    private:
        [[nodiscard]] cuda::Buffer<float> make_buffer(std::size_t size) const;
        [[nodiscard]] cuda::Buffer<double> make_adjoint_buffer(std::size_t size) const;
        [[nodiscard]] cuda::Buffer<std::uint32_t> make_index_buffer(std::size_t size) const;
        [[nodiscard]] ScalarField make_scalar_field() const;
        [[nodiscard]] CenteredVectorField make_centered_vector_field() const;
        [[nodiscard]] StaggeredVectorField make_staggered_vector_field() const;
        [[nodiscard]] VorticityCache make_vorticity_cache() const;
        [[nodiscard]] ScalarAdjointField make_scalar_adjoint_field() const;
        [[nodiscard]] CenteredVectorAdjointField make_centered_vector_adjoint_field() const;
        [[nodiscard]] StaggeredVectorAdjointField make_staggered_vector_adjoint_field() const;
        [[nodiscard]] VorticityAdjointCache make_vorticity_adjoint_cache() const;
        void zero(cuda::Buffer<float>& buffer);
        void zero(cuda::Buffer<double>& buffer);
        void zero(ScalarField& field);
        void zero(CenteredVectorField& field);
        void zero(StaggeredVectorField& field);
        void zero(VorticityCache& cache);
        void zero(ScalarAdjointField& field);
        void zero(CenteredVectorAdjointField& field);
        void zero(StaggeredVectorAdjointField& field);
        void zero(VorticityAdjointCache& cache);
        void copy(const cuda::Buffer<float>& source, cuda::Buffer<float>& destination);
        void copy(const cuda::Buffer<double>& source, cuda::Buffer<double>& destination);
        void copy(const ScalarField& source, ScalarField& destination);
        void copy(const StaggeredVectorField& source, StaggeredVectorField& destination);
        void copy(const ScalarAdjointField& source, ScalarAdjointField& destination);
        void copy(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination);
        void accumulate(const ScalarAdjointField& source, ScalarAdjointField& destination);
        void accumulate(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination);

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
        const Configuration configuration;

        explicit Model(Configuration configuration);

        [[nodiscard]] ExecutionContext make_context(ExecutionMode mode) const;
        [[nodiscard]] State make_state(ExecutionContext& context) const;
        [[nodiscard]] Control make_control(ExecutionContext& context) const;
        [[nodiscard]] Parameters make_parameters(ExecutionContext& context) const;
        [[nodiscard]] StepCache make_step_cache(ExecutionContext& context) const;
        [[nodiscard]] VorticityAdjointCache make_vorticity_adjoint_cache(ExecutionContext& context) const;
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
        SourceOperator source_;
        ForceAssemblyOperator force_;
        VelocityEvolutionOperator velocity_;
        PressureProjectionOperator projection_;
        ScalarAdvectionOperator scalar_advection_;
    };

} // namespace xayah::smoke
