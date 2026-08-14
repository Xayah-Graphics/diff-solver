export module xayah.fluid.dfsph;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.sph;

export namespace xayah::fluid::dfsph {

    struct State : fluid::State {
        fluid::ScalarField warm_divergence_pressure;
        fluid::ScalarField warm_density_pressure;
    };

    struct StateTangent : fluid::StateTangent {
        fluid::ScalarField warm_divergence_pressure;
        fluid::ScalarField warm_density_pressure;
    };

    struct StateAdjoint : fluid::StateAdjoint {
        fluid::ScalarAdjointField warm_divergence_pressure;
        fluid::ScalarAdjointField warm_density_pressure;
    };

    struct Configuration {
        fluid::Configuration fluid;
        std::uint32_t divergence_iterations{4u};
        std::uint32_t density_iterations{6u};
        std::uint32_t checkpoint_interval{2u};
        bool pressure_warm_start{true};
    };

    struct Parameters {
        fluid::ParticleParameters particles;
        cuda::Buffer<float> divergence_relaxation;
        cuda::Buffer<float> density_relaxation;
    };

    struct ParameterTangent {
        fluid::ParticleParameterTangent particles;
        cuda::Buffer<float> divergence_relaxation;
        cuda::Buffer<float> density_relaxation;
    };

    struct ParameterAdjoint {
        fluid::ParticleParameterAdjoint particles;
        cuda::Buffer<double> divergence_relaxation;
        cuda::Buffer<double> density_relaxation;
    };

    struct ProjectionIterationCache {
        std::uint32_t iteration;
        fluid::ScalarField pressure_impulses;
        fluid::ScalarField predicted_densities;
        fluid::VectorField pressure_accelerations;
        fluid::VectorField predicted_positions;
        fluid::VectorField predicted_velocities;
    };

    struct ProjectionIterationTangent {
        fluid::ScalarField pressure_impulses;
        fluid::ScalarField predicted_densities;
        fluid::VectorField pressure_accelerations;
        fluid::VectorField predicted_positions;
        fluid::VectorField predicted_velocities;
    };

    struct ProjectionIterationAdjoint {
        fluid::ScalarAdjointField pressure_impulses;
        fluid::ScalarAdjointField predicted_densities;
        fluid::VectorAdjointField pressure_accelerations;
        fluid::VectorAdjointField predicted_positions;
        fluid::VectorAdjointField predicted_velocities;
    };

    struct StepCache {
        fluid::Neighborhood neighborhood;
        fluid::ScalarField densities;
        fluid::VectorField non_pressure_accelerations;
        std::vector<ProjectionIterationCache> divergence_checkpoints;
        fluid::VectorField divergence_pressure_accelerations;
        std::vector<ProjectionIterationCache> density_checkpoints;
        fluid::VectorField total_pressure_accelerations;
    };

    struct Model;

    struct ExecutionContext {
        std::shared_ptr<cuda::Resource> resource;
        fluid::NeighborSearch neighbor_search;
        fluid::sph::DeviceBoundary boundary;

    private:
        ProjectionIterationCache primal_scratch_;
        std::vector<ProjectionIterationCache> recomputed_iterations_;
        fluid::VectorField total_pressure_acceleration_;
        ProjectionIterationTangent tangent_scratch_;
        ProjectionIterationAdjoint adjoint_scratch_;
        ProjectionIterationAdjoint previous_adjoint_scratch_;
        fluid::ScalarField density_tangent_;
        fluid::VectorField non_pressure_acceleration_tangent_;
        fluid::VectorField divergence_pressure_acceleration_tangent_;
        fluid::VectorField total_pressure_acceleration_tangent_;
        fluid::ScalarAdjointField density_adjoint_;
        fluid::ScalarAdjointField target_density_adjoint_;
        fluid::VectorAdjointField non_pressure_acceleration_adjoint_;
        fluid::VectorAdjointField divergence_pressure_acceleration_adjoint_;
        fluid::VectorAdjointField total_pressure_acceleration_adjoint_;

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
        float reference_gradient_norm_;
    };

} // namespace xayah::fluid::dfsph
