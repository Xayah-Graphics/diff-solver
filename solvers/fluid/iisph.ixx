export module xayah.fluid.iisph;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.sph;

export namespace xayah::fluid::iisph {

    struct Configuration {
        fluid::Configuration fluid;
        std::uint32_t pressure_iterations{6u};
        std::uint32_t checkpoint_interval{2u};
    };

    struct Parameters {
        fluid::ParticleParameters particles;
        cuda::Buffer<float> jacobi_relaxation;
    };

    struct ParameterTangent {
        fluid::ParticleParameterTangent particles;
        cuda::Buffer<float> jacobi_relaxation;
    };

    struct ParameterAdjoint {
        fluid::ParticleParameterAdjoint particles;
        cuda::Buffer<double> jacobi_relaxation;
    };

    struct IterationCache {
        std::uint32_t iteration;
        fluid::ScalarField pressures;
        fluid::ScalarField predicted_densities;
        fluid::VectorField pressure_accelerations;
        fluid::VectorField predicted_positions;
        fluid::VectorField predicted_velocities;
    };

    struct IterationTangent {
        fluid::ScalarField pressures;
        fluid::ScalarField predicted_densities;
        fluid::VectorField pressure_accelerations;
        fluid::VectorField predicted_positions;
        fluid::VectorField predicted_velocities;
    };

    struct IterationAdjoint {
        fluid::ScalarAdjointField pressures;
        fluid::ScalarAdjointField predicted_densities;
        fluid::VectorAdjointField pressure_accelerations;
        fluid::VectorAdjointField predicted_positions;
        fluid::VectorAdjointField predicted_velocities;
    };

    struct StepCache {
        fluid::Neighborhood neighborhood;
        fluid::ScalarField densities;
        fluid::VectorField non_pressure_accelerations;
        std::vector<IterationCache> checkpoints;
    };

    struct Model;

    struct ExecutionContext {
        std::shared_ptr<cuda::Resource> resource;
        fluid::NeighborSearch neighbor_search;
        fluid::sph::DeviceBoundary boundary;

    private:
        IterationCache primal_scratch_;
        std::vector<IterationCache> recomputed_iterations_;
        IterationTangent tangent_scratch_;
        IterationAdjoint adjoint_scratch_;
        IterationAdjoint previous_adjoint_scratch_;
        fluid::ScalarField density_tangent_;
        fluid::VectorField non_pressure_acceleration_tangent_;
        fluid::ScalarAdjointField density_adjoint_;
        fluid::VectorAdjointField non_pressure_acceleration_adjoint_;

        friend struct Model;
    };

    struct Model {
        const Configuration configuration;

        explicit Model(Configuration configuration);

        [[nodiscard]] ExecutionContext allocate_context(fluid::ExecutionMode mode) const;
        [[nodiscard]] fluid::State allocate_state(ExecutionContext& context) const;
        [[nodiscard]] fluid::Control allocate_control(ExecutionContext& context) const;
        [[nodiscard]] Parameters allocate_parameters(ExecutionContext& context) const;
        [[nodiscard]] StepCache allocate_step_cache(ExecutionContext& context) const;
        [[nodiscard]] fluid::StateTangent allocate_state_tangent(ExecutionContext& context) const;
        [[nodiscard]] fluid::ControlTangent allocate_control_tangent(ExecutionContext& context) const;
        [[nodiscard]] ParameterTangent allocate_parameter_tangent(ExecutionContext& context) const;
        [[nodiscard]] fluid::StateAdjoint allocate_state_adjoint(ExecutionContext& context) const;
        [[nodiscard]] fluid::ControlAdjoint allocate_control_adjoint(ExecutionContext& context) const;
        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(ExecutionContext& context) const;

        void copy_state(const fluid::State& source, fluid::State& destination, ExecutionContext& context) const;
        void copy_state_tangent(const fluid::StateTangent& source, fluid::StateTangent& destination, ExecutionContext& context) const;
        void copy_state_adjoint(const fluid::StateAdjoint& source, fluid::StateAdjoint& destination, ExecutionContext& context) const;
        void accumulate_state_adjoint(const fluid::StateAdjoint& source, fluid::StateAdjoint& destination, ExecutionContext& context) const;

        void forward_step(const fluid::State& state, const fluid::Control& control, const Parameters& parameters, fluid::State& next_state, StepCache& step_cache, ExecutionContext& context) const;
        void jvp_step(const fluid::State& state, const fluid::Control& control, const Parameters& parameters, const fluid::State& next_state, const StepCache& step_cache, const fluid::StateTangent& state_tangent, const fluid::ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, fluid::StateTangent& next_state_tangent, ExecutionContext& context) const;
        void vjp_step(const fluid::State& state, const fluid::Control& control, const Parameters& parameters, const fluid::State& next_state, const StepCache& step_cache, const fluid::StateAdjoint& next_state_adjoint, fluid::StateAdjoint& previous_state_adjoint, fluid::ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const;

    private:
        float reference_gradient_norm_;
    };

} // namespace xayah::fluid::iisph
