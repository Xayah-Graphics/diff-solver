export module xayah.fluid.wcsph;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.sph;

export namespace xayah::fluid::wcsph {

    struct Configuration {
        fluid::Configuration fluid;
    };

    struct Parameters {
        fluid::ParticleParameters particles;
        cuda::Buffer<float> speed_of_sound;
        cuda::Buffer<float> tait_exponent;
        cuda::Buffer<float> boundary_surface_tension;
    };

    struct ParameterTangent {
        fluid::ParticleParameterTangent particles;
        cuda::Buffer<float> speed_of_sound;
        cuda::Buffer<float> tait_exponent;
        cuda::Buffer<float> boundary_surface_tension;
    };

    struct ParameterAdjoint {
        fluid::ParticleParameterAdjoint particles;
        cuda::Buffer<double> speed_of_sound;
        cuda::Buffer<double> tait_exponent;
        cuda::Buffer<double> boundary_surface_tension;
    };

    struct StepCache {
        fluid::Neighborhood neighborhood;
        fluid::ScalarField densities;
        fluid::ScalarField pressures;
        fluid::VectorField pressure_accelerations;
        fluid::VectorField viscosity_accelerations;
        fluid::VectorField surface_accelerations;
        fluid::VectorField external_accelerations;
        fluid::VectorField total_accelerations;
    };

    struct Model;

    struct ExecutionContext {
        std::shared_ptr<cuda::Resource> resource;
        fluid::NeighborSearch neighbor_search;
        fluid::sph::DeviceBoundary boundary;

    private:
        fluid::ScalarField density_tangent_;
        fluid::ScalarField pressure_tangent_;
        fluid::VectorField pressure_acceleration_tangent_;
        fluid::VectorField viscosity_acceleration_tangent_;
        fluid::VectorField surface_acceleration_tangent_;
        fluid::VectorField total_acceleration_tangent_;
        fluid::ScalarAdjointField density_adjoint_;
        fluid::ScalarAdjointField pressure_adjoint_;
        fluid::VectorAdjointField pressure_acceleration_adjoint_;
        fluid::VectorAdjointField viscosity_acceleration_adjoint_;
        fluid::VectorAdjointField surface_acceleration_adjoint_;
        fluid::VectorAdjointField total_acceleration_adjoint_;

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

} // namespace xayah::fluid::wcsph
