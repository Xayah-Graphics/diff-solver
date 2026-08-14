export module xayah.fluid.pbf;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.sph;

export namespace xayah::fluid::pbf {

    struct Configuration {
        fluid::Configuration fluid;
        std::uint32_t pressure_iterations{5u};
        std::uint32_t checkpoint_interval{2u};
    };

    struct Parameters {
        fluid::ParticleParameters particles;
        cuda::Buffer<float> relaxation;
        cuda::Buffer<float> artificial_pressure_strength;
        cuda::Buffer<float> artificial_pressure_exponent;
        cuda::Buffer<float> artificial_pressure_radius;
        cuda::Buffer<float> xsph_viscosity;
        cuda::Buffer<float> vorticity_confinement;
    };

    struct ParameterTangent {
        fluid::ParticleParameterTangent particles;
        cuda::Buffer<float> relaxation;
        cuda::Buffer<float> artificial_pressure_strength;
        cuda::Buffer<float> artificial_pressure_exponent;
        cuda::Buffer<float> artificial_pressure_radius;
        cuda::Buffer<float> xsph_viscosity;
        cuda::Buffer<float> vorticity_confinement;
    };

    struct ParameterAdjoint {
        fluid::ParticleParameterAdjoint particles;
        cuda::Buffer<double> relaxation;
        cuda::Buffer<double> artificial_pressure_strength;
        cuda::Buffer<double> artificial_pressure_exponent;
        cuda::Buffer<double> artificial_pressure_radius;
        cuda::Buffer<double> xsph_viscosity;
        cuda::Buffer<double> vorticity_confinement;
    };

    struct StepCache {
        fluid::Neighborhood neighborhood;
        fluid::VectorField predicted_positions;
        fluid::VectorField corrected_positions;
        struct IterationCheckpoint {
            std::uint32_t iteration;
            fluid::VectorField positions;
        };
        std::vector<IterationCheckpoint> checkpoints;
        fluid::VectorField reconstructed_velocities;
        fluid::VectorField vorticities;
        fluid::ScalarField vorticity_magnitudes;
        fluid::VectorField vorticity_normals;
        fluid::ScalarField vorticity_normalizers;
        fluid::VectorField confined_velocities;
    };

    struct Model;

    struct ExecutionContext {
        std::shared_ptr<cuda::Resource> resource;
        fluid::NeighborSearch neighbor_search;
        fluid::sph::DeviceBoundary boundary;

    private:
        struct IterationPrimal {
            fluid::ScalarField densities;
            fluid::VectorField gradient_sums;
            fluid::ScalarField denominators;
            fluid::ScalarField lambdas;
            fluid::VectorField corrections;
            cuda::Buffer<std::uint32_t> collision_masks;
        };

        IterationPrimal forward_iteration_;
        std::vector<fluid::VectorField> segment_position_history_;
        std::vector<IterationPrimal> segment_iteration_history_;
        fluid::VectorField current_position_tangent_;
        fluid::VectorField next_position_tangent_;
        fluid::ScalarField density_tangent_;
        fluid::VectorField gradient_sum_tangent_;
        fluid::ScalarField denominator_tangent_;
        fluid::ScalarField lambda_tangent_;
        fluid::VectorField correction_tangent_;
        fluid::VectorField reconstructed_velocity_tangent_;
        fluid::VectorField vorticity_tangent_;
        fluid::ScalarField vorticity_magnitude_tangent_;
        fluid::VectorField vorticity_normal_tangent_;
        fluid::VectorField confined_velocity_tangent_;
        fluid::VectorAdjointField current_position_adjoint_;
        fluid::VectorAdjointField next_position_adjoint_;
        fluid::VectorAdjointField correction_adjoint_;
        fluid::ScalarAdjointField lambda_adjoint_;
        fluid::ScalarAdjointField density_adjoint_;
        fluid::VectorAdjointField reconstructed_velocity_adjoint_;
        fluid::VectorAdjointField vorticity_adjoint_;
        fluid::ScalarAdjointField vorticity_magnitude_adjoint_;
        fluid::VectorAdjointField vorticity_normal_adjoint_;
        fluid::VectorAdjointField confined_velocity_adjoint_;

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
    };

} // namespace xayah::fluid::pbf
