export module xayah.fluid.apic;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.grid;

export namespace xayah::fluid::apic {

    struct Configuration {
        std::uint32_t particle_count;
        float particle_radius;
        fluid::Vector3 gravity;
        grid::Configuration grid;
    };

    struct State {
        fluid::VectorField positions;
        fluid::VectorField velocities;
        fluid::MatrixField affine;
        std::uint64_t step_index{};
    };

    struct StateTangent {
        fluid::VectorField positions;
        fluid::VectorField velocities;
        fluid::MatrixField affine;
    };

    struct StateAdjoint {
        fluid::VectorAdjointField positions;
        fluid::VectorAdjointField velocities;
        fluid::MatrixAdjointField affine;
    };

    struct Parameters {
        cuda::Buffer<float> masses;
    };

    struct ParameterTangent {
        cuda::Buffer<float> masses;
    };

    struct ParameterAdjoint {
        cuda::Buffer<double> masses;
    };

    struct StepCache {
        grid::StepCache grid;
        fluid::VectorField transferred_velocities;
    };

    struct Model;

    struct ExecutionContext {
        std::shared_ptr<cuda::Resource> resource;
        grid::DeviceDomain domain;

    private:
        grid::DifferentialScratch grid_differential_scratch_;
        fluid::VectorField transferred_velocity_tangent_;
        fluid::VectorAdjointField transferred_velocity_adjoint_;

        friend struct Model;
    };

    struct Model {
        const Configuration configuration;

        explicit Model(Configuration configuration);

        [[nodiscard]] ExecutionContext allocate_context(fluid::ExecutionMode mode) const;
        [[nodiscard]] State allocate_state(ExecutionContext& context) const;
        [[nodiscard]] fluid::Control allocate_control(ExecutionContext& context) const;
        [[nodiscard]] Parameters allocate_parameters(ExecutionContext& context) const;
        [[nodiscard]] StepCache allocate_step_cache(ExecutionContext& context) const;
        [[nodiscard]] StateTangent allocate_state_tangent(ExecutionContext& context) const;
        [[nodiscard]] fluid::ControlTangent allocate_control_tangent(ExecutionContext& context) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(ExecutionContext& context) const;
        [[nodiscard]] StateAdjoint allocate_state_adjoint(ExecutionContext& context) const;
        [[nodiscard]] fluid::ControlAdjoint allocate_control_adjoint(ExecutionContext& context) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(ExecutionContext& context) const;

        void copy_state(const State& source, State& destination, ExecutionContext& context) const;
        void copy_state_tangent(const StateTangent& source, StateTangent& destination, ExecutionContext& context) const;
        void copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const;
        void accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const;

        void forward_step(const State& state, const fluid::Control& control, const Parameters& parameters, State& next_state, StepCache& step_cache, ExecutionContext& context) const;
        void jvp_step(const State& state, const fluid::Control& control, const Parameters& parameters, const State& next_state, const StepCache& step_cache, const StateTangent& state_tangent, const fluid::ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const;
        void vjp_step(const State& state, const fluid::Control& control, const Parameters& parameters, const State& next_state, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, fluid::ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const;

    private:
        grid::MarkerOperator marker_;
        grid::ParticleToGridOperator particle_to_grid_;
        grid::ProjectionOperator projection_;
        grid::ExtrapolationOperator extrapolation_;
        grid::GridToParticleOperator grid_to_particle_;
        grid::ParticleAdvectionOperator advection_;
    };

} // namespace xayah::fluid::apic
