module;

#include "iisph.h"

module xayah.fluid.iisph;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.sph;

namespace xayah::fluid::iisph {

    namespace {

        sph::cuda_kernel::ConstVector kernel_vector(const VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        sph::cuda_kernel::Vector kernel_vector(VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        sph::cuda_kernel::ConstVectorAdjoint kernel_vector(const VectorAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        sph::cuda_kernel::VectorAdjoint kernel_vector(VectorAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        sph::cuda_kernel::ParticleParameters kernel_parameters(const ParticleParameters& parameters) {
            return {.masses = parameters.masses.data, .rest_densities = parameters.rest_densities.data, .viscosities = parameters.viscosities.data, .surface_tensions = parameters.surface_tensions.data};
        }

        sph::cuda_kernel::ParticleParameterTangent kernel_parameters(const ParticleParameterTangent& parameters) {
            return {.masses = parameters.masses.data, .rest_densities = parameters.rest_densities.data, .viscosities = parameters.viscosities.data, .surface_tensions = parameters.surface_tensions.data};
        }

        sph::cuda_kernel::ParticleParameterAdjoint kernel_parameters(ParticleParameterAdjoint& parameters) {
            return {.masses = parameters.masses.data, .rest_densities = parameters.rest_densities.data, .viscosities = parameters.viscosities.data, .surface_tensions = parameters.surface_tensions.data};
        }

        sph::cuda_kernel::Domain collision_domain(const fluid::Configuration& configuration, const std::uint64_t step_index) {
            const float time = static_cast<float>(step_index) * configuration.time_step;
            return {
                .minimum_x = configuration.domain.minimum.x + time * configuration.domain.velocity.x + configuration.particle_radius,
                .minimum_y = configuration.domain.minimum.y + time * configuration.domain.velocity.y + configuration.particle_radius,
                .minimum_z = configuration.domain.minimum.z + time * configuration.domain.velocity.z + configuration.particle_radius,
                .maximum_x = configuration.domain.maximum.x + time * configuration.domain.velocity.x - configuration.particle_radius,
                .maximum_y = configuration.domain.maximum.y + time * configuration.domain.velocity.y - configuration.particle_radius,
                .maximum_z = configuration.domain.maximum.z + time * configuration.domain.velocity.z - configuration.particle_radius,
                .velocity_x = configuration.domain.velocity.x,
                .velocity_y = configuration.domain.velocity.y,
                .velocity_z = configuration.domain.velocity.z,
                .no_slip = configuration.domain.no_slip ? 1u : 0u,
            };
        }

        IterationCache allocate_iteration_cache(const std::shared_ptr<cuda::Resource>& resource, const std::uint32_t count) {
            return {
                .iteration = 0u,
                .pressures = sph::allocate_scalar_field(resource, count),
                .predicted_densities = sph::allocate_scalar_field(resource, count),
                .pressure_accelerations = sph::allocate_vector_field(resource, count),
                .predicted_positions = sph::allocate_vector_field(resource, count),
                .predicted_velocities = sph::allocate_vector_field(resource, count),
            };
        }

        IterationTangent allocate_iteration_tangent(const std::shared_ptr<cuda::Resource>& resource, const std::uint32_t count) {
            return {
                .pressures = sph::allocate_scalar_field(resource, count),
                .predicted_densities = sph::allocate_scalar_field(resource, count),
                .pressure_accelerations = sph::allocate_vector_field(resource, count),
                .predicted_positions = sph::allocate_vector_field(resource, count),
                .predicted_velocities = sph::allocate_vector_field(resource, count),
            };
        }

        IterationAdjoint allocate_iteration_adjoint(const std::shared_ptr<cuda::Resource>& resource, const std::uint32_t count) {
            return {
                .pressures = sph::allocate_scalar_adjoint_field(resource, count),
                .predicted_densities = sph::allocate_scalar_adjoint_field(resource, count),
                .pressure_accelerations = sph::allocate_vector_adjoint_field(resource, count),
                .predicted_positions = sph::allocate_vector_adjoint_field(resource, count),
                .predicted_velocities = sph::allocate_vector_adjoint_field(resource, count),
            };
        }

        void zero_iteration(cuda::Resource& resource, IterationCache& cache) {
            sph::zero_scalar(resource, cache.pressures);
            sph::zero_scalar(resource, cache.predicted_densities);
            sph::zero_vector(resource, cache.pressure_accelerations);
            sph::zero_vector(resource, cache.predicted_positions);
            sph::zero_vector(resource, cache.predicted_velocities);
        }

        void zero_iteration(cuda::Resource& resource, IterationTangent& tangent) {
            sph::zero_scalar(resource, tangent.pressures);
            sph::zero_scalar(resource, tangent.predicted_densities);
            sph::zero_vector(resource, tangent.pressure_accelerations);
            sph::zero_vector(resource, tangent.predicted_positions);
            sph::zero_vector(resource, tangent.predicted_velocities);
        }

        void zero_iteration(cuda::Resource& resource, IterationAdjoint& adjoint) {
            sph::zero_scalar_adjoint(resource, adjoint.pressures);
            sph::zero_scalar_adjoint(resource, adjoint.predicted_densities);
            sph::zero_vector_adjoint(resource, adjoint.pressure_accelerations);
            sph::zero_vector_adjoint(resource, adjoint.predicted_positions);
            sph::zero_vector_adjoint(resource, adjoint.predicted_velocities);
        }

        void copy_iteration(cuda::Resource& resource, const IterationCache& source, IterationCache& destination, const std::uint32_t count) {
            destination.iteration = source.iteration;
            cuda_kernel::launch_copy_iteration_forward(resource.native_stream, count, source.pressures.values.data, source.predicted_densities.values.data, kernel_vector(source.pressure_accelerations), kernel_vector(source.predicted_positions), kernel_vector(source.predicted_velocities), destination.pressures.values.data, destination.predicted_densities.values.data, kernel_vector(destination.pressure_accelerations), kernel_vector(destination.predicted_positions), kernel_vector(destination.predicted_velocities));
        }

        void zero_parameters(cuda::Resource& resource, ParameterTangent& parameters) {
            resource.zero(parameters.particles.masses.data, parameters.particles.masses.size * sizeof(float));
            resource.zero(parameters.particles.rest_densities.data, parameters.particles.rest_densities.size * sizeof(float));
            resource.zero(parameters.particles.viscosities.data, parameters.particles.viscosities.size * sizeof(float));
            resource.zero(parameters.particles.surface_tensions.data, parameters.particles.surface_tensions.size * sizeof(float));
            resource.zero(parameters.jacobi_relaxation.data, parameters.jacobi_relaxation.size * sizeof(float));
        }

        void zero_parameters(cuda::Resource& resource, ParameterAdjoint& parameters) {
            resource.zero(parameters.particles.masses.data, parameters.particles.masses.size * sizeof(double));
            resource.zero(parameters.particles.rest_densities.data, parameters.particles.rest_densities.size * sizeof(double));
            resource.zero(parameters.particles.viscosities.data, parameters.particles.viscosities.size * sizeof(double));
            resource.zero(parameters.particles.surface_tensions.data, parameters.particles.surface_tensions.size * sizeof(double));
            resource.zero(parameters.jacobi_relaxation.data, parameters.jacobi_relaxation.size * sizeof(double));
        }

    } // namespace

    Model::Model(Configuration next_configuration) : configuration(std::move(next_configuration)) {
        constexpr float pi = 3.14159265358979323846F;
        const float support_radius = configuration.fluid.support_radius;
        const float diameter = 2.0F * configuration.fluid.particle_radius;
        const float coefficient = 8.0F / (pi * support_radius * support_radius * support_radius);
        Vector3 gradient_sum{};
        float squared_gradient_sum = 0.0F;
        for (float x = -support_radius; x <= support_radius; x += diameter)
            for (float y = -support_radius; y <= support_radius; y += diameter)
                for (float z = -support_radius; z <= support_radius; z += diameter) {
                    const float distance = std::sqrt(x * x + y * y + z * z);
                    if (distance == 0.0F || distance >= support_radius) continue;
                    const float q = distance / support_radius;
                    const float derivative = q <= 0.5F ? 18.0F * q * q - 12.0F * q : -6.0F * (1.0F - q) * (1.0F - q);
                    const float scale = coefficient * derivative / (support_radius * distance);
                    gradient_sum.x += scale * x;
                    gradient_sum.y += scale * y;
                    gradient_sum.z += scale * z;
                    squared_gradient_sum += scale * scale * (x * x + y * y + z * z);
                }
        reference_gradient_norm_ = gradient_sum.x * gradient_sum.x + gradient_sum.y * gradient_sum.y + gradient_sum.z * gradient_sum.z + squared_gradient_sum;
    }

    ExecutionContext Model::allocate_context(const ExecutionMode mode) const {
        ExecutionContext context{};
        context.resource = std::make_shared<cuda::Resource>();
        context.neighbor_search = sph::allocate_neighbor_search(context.resource, configuration.fluid);
        context.boundary = sph::allocate_boundary(context.resource, configuration.fluid);
        context.primal_scratch_ = allocate_iteration_cache(context.resource, configuration.fluid.particle_count);
        context.recomputed_iterations_.reserve(configuration.checkpoint_interval + 1u);
        for (std::uint32_t iteration = 0u; iteration <= configuration.checkpoint_interval; ++iteration)
            context.recomputed_iterations_.push_back(allocate_iteration_cache(context.resource, configuration.fluid.particle_count));
        if (mode == ExecutionMode::differentiable) {
            context.tangent_scratch_ = allocate_iteration_tangent(context.resource, configuration.fluid.particle_count);
            context.adjoint_scratch_ = allocate_iteration_adjoint(context.resource, configuration.fluid.particle_count);
            context.previous_adjoint_scratch_ = allocate_iteration_adjoint(context.resource, configuration.fluid.particle_count);
            context.density_tangent_ = sph::allocate_scalar_field(context.resource, configuration.fluid.particle_count);
            context.non_pressure_acceleration_tangent_ = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count);
            context.density_adjoint_ = sph::allocate_scalar_adjoint_field(context.resource, configuration.fluid.particle_count);
            context.non_pressure_acceleration_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, configuration.fluid.particle_count);
        }
        return context;
    }

    State Model::allocate_state(ExecutionContext& context) const {
        return sph::allocate_state(context.resource, configuration.fluid.particle_count);
    }

    Control Model::allocate_control(ExecutionContext& context) const {
        return sph::allocate_control(context.resource, configuration.fluid.particle_count);
    }

    Parameters Model::allocate_parameters(ExecutionContext& context) const {
        return {
            .particles = sph::allocate_particle_parameters(context.resource, configuration.fluid.particle_count),
            .jacobi_relaxation = cuda::Buffer<float>(context.resource, configuration.fluid.particle_count),
        };
    }

    StepCache Model::allocate_step_cache(ExecutionContext& context) const {
        StepCache cache{
            .neighborhood = sph::allocate_neighborhood(context.resource, configuration.fluid),
            .densities = sph::allocate_scalar_field(context.resource, configuration.fluid.particle_count),
            .non_pressure_accelerations = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count),
            .checkpoints = {},
        };
        const std::uint32_t checkpoint_count = 1u + configuration.pressure_iterations / configuration.checkpoint_interval + (configuration.pressure_iterations % configuration.checkpoint_interval == 0u ? 0u : 1u);
        cache.checkpoints.reserve(checkpoint_count);
        for (std::uint32_t checkpoint = 0u; checkpoint < checkpoint_count; ++checkpoint)
            cache.checkpoints.push_back(allocate_iteration_cache(context.resource, configuration.fluid.particle_count));
        return cache;
    }

    StateTangent Model::allocate_state_tangent(ExecutionContext& context) const {
        return sph::allocate_state_tangent(context.resource, configuration.fluid.particle_count);
    }

    ControlTangent Model::allocate_control_tangent(ExecutionContext& context) const {
        return sph::allocate_control_tangent(context.resource, configuration.fluid.particle_count);
    }

    ParameterTangent Model::allocate_parameter_tangent(ExecutionContext& context) const {
        ParameterTangent tangent{
            .particles = sph::allocate_particle_parameter_tangent(context.resource, configuration.fluid.particle_count),
            .jacobi_relaxation = cuda::Buffer<float>(context.resource, configuration.fluid.particle_count),
        };
        zero_parameters(*context.resource, tangent);
        return tangent;
    }

    StateAdjoint Model::allocate_state_adjoint(ExecutionContext& context) const {
        return sph::allocate_state_adjoint(context.resource, configuration.fluid.particle_count);
    }

    ControlAdjoint Model::allocate_control_adjoint(ExecutionContext& context) const {
        return sph::allocate_control_adjoint(context.resource, configuration.fluid.particle_count);
    }

    ParameterAdjoint Model::allocate_parameter_adjoint(ExecutionContext& context) const {
        ParameterAdjoint adjoint{
            .particles = sph::allocate_particle_parameter_adjoint(context.resource, configuration.fluid.particle_count),
            .jacobi_relaxation = cuda::Buffer<double>(context.resource, configuration.fluid.particle_count),
        };
        zero_parameters(*context.resource, adjoint);
        return adjoint;
    }

    void Model::copy_state(const State& source, State& destination, ExecutionContext& context) const {
        sph::copy_state(*context.resource, source, destination);
    }

    void Model::copy_state_tangent(const StateTangent& source, StateTangent& destination, ExecutionContext& context) const {
        sph::copy_state_tangent(*context.resource, source, destination);
    }

    void Model::copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        sph::copy_state_adjoint(*context.resource, source, destination);
    }

    void Model::accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        sph::accumulate_state_adjoint(*context.resource, source, destination);
    }

    void Model::forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& step_cache, ExecutionContext& context) const {
        sph::build_neighborhood(*context.resource, configuration.fluid, state.step_index, context.boundary, state.positions, context.neighbor_search, step_cache.neighborhood);
        sph::density_forward(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities);
        sph::non_pressure_forward(*context.resource, configuration.fluid, context.boundary, state, control, parameters.particles, step_cache.neighborhood, step_cache.densities, step_cache.non_pressure_accelerations);
        zero_iteration(*context.resource, context.primal_scratch_);
        context.primal_scratch_.iteration = 0u;
        copy_iteration(*context.resource, context.primal_scratch_, step_cache.checkpoints[0], configuration.fluid.particle_count);
        std::uint32_t checkpoint = 1u;
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            cuda_kernel::launch_predict_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, collision_domain(configuration.fluid, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(step_cache.non_pressure_accelerations), kernel_vector(context.primal_scratch_.pressure_accelerations), kernel_vector(context.primal_scratch_.predicted_positions), kernel_vector(context.primal_scratch_.predicted_velocities));
            sph::density_forward_frozen(*context.resource, configuration.fluid, context.boundary, state.positions, context.primal_scratch_.predicted_positions, parameters.particles, step_cache.neighborhood, context.primal_scratch_.predicted_densities);
            cuda_kernel::launch_jacobi_update_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, reference_gradient_norm_, kernel_parameters(parameters.particles), step_cache.densities.values.data, context.primal_scratch_.pressures.values.data, context.primal_scratch_.predicted_densities.values.data, parameters.jacobi_relaxation.data, context.primal_scratch_.pressures.values.data);
            sph::pressure_acceleration_forward(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities, context.primal_scratch_.pressures, context.primal_scratch_.pressure_accelerations);
            context.primal_scratch_.iteration = iteration;
            if (iteration % configuration.checkpoint_interval == 0u || iteration == configuration.pressure_iterations) copy_iteration(*context.resource, context.primal_scratch_, step_cache.checkpoints[checkpoint++], configuration.fluid.particle_count);
        }
        cuda_kernel::launch_predict_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, collision_domain(configuration.fluid, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(step_cache.non_pressure_accelerations), kernel_vector(context.primal_scratch_.pressure_accelerations), kernel_vector(next_state.positions), kernel_vector(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void Model::jvp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const {
        sph::density_jvp(*context.resource, configuration.fluid, context.boundary, state.positions, state_tangent.positions, parameters.particles, parameter_tangent.particles, step_cache.neighborhood, context.density_tangent_);
        sph::non_pressure_jvp(*context.resource, configuration.fluid, context.boundary, state, parameters.particles, step_cache.neighborhood, step_cache.densities, state_tangent, control_tangent, parameter_tangent.particles, context.density_tangent_, context.non_pressure_acceleration_tangent_);
        zero_iteration(*context.resource, context.primal_scratch_);
        zero_iteration(*context.resource, context.tangent_scratch_);
        for (std::uint32_t iteration = 1u; iteration <= configuration.pressure_iterations; ++iteration) {
            cuda_kernel::launch_predict_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, collision_domain(configuration.fluid, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(step_cache.non_pressure_accelerations), kernel_vector(context.primal_scratch_.pressure_accelerations), kernel_vector(context.primal_scratch_.predicted_positions), kernel_vector(context.primal_scratch_.predicted_velocities));
            cuda_kernel::launch_predict_jvp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, collision_domain(configuration.fluid, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(step_cache.non_pressure_accelerations), kernel_vector(context.primal_scratch_.pressure_accelerations), kernel_vector(state_tangent.positions), kernel_vector(state_tangent.velocities), kernel_vector(context.non_pressure_acceleration_tangent_), kernel_vector(context.tangent_scratch_.pressure_accelerations), kernel_vector(context.tangent_scratch_.predicted_positions), kernel_vector(context.tangent_scratch_.predicted_velocities));
            sph::density_forward_frozen(*context.resource, configuration.fluid, context.boundary, state.positions, context.primal_scratch_.predicted_positions, parameters.particles, step_cache.neighborhood, context.primal_scratch_.predicted_densities);
            sph::density_jvp_frozen(*context.resource, configuration.fluid, context.boundary, state.positions, context.primal_scratch_.predicted_positions, context.tangent_scratch_.predicted_positions, parameters.particles, parameter_tangent.particles, step_cache.neighborhood, context.tangent_scratch_.predicted_densities);
            cuda_kernel::launch_jacobi_update_jvp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, reference_gradient_norm_, kernel_parameters(parameters.particles), kernel_parameters(parameter_tangent.particles), step_cache.densities.values.data, context.density_tangent_.values.data, context.primal_scratch_.pressures.values.data, context.primal_scratch_.predicted_densities.values.data, parameters.jacobi_relaxation.data, context.tangent_scratch_.pressures.values.data, context.tangent_scratch_.predicted_densities.values.data, parameter_tangent.jacobi_relaxation.data, context.tangent_scratch_.pressures.values.data);
            cuda_kernel::launch_jacobi_update_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, reference_gradient_norm_, kernel_parameters(parameters.particles), step_cache.densities.values.data, context.primal_scratch_.pressures.values.data, context.primal_scratch_.predicted_densities.values.data, parameters.jacobi_relaxation.data, context.primal_scratch_.pressures.values.data);
            sph::pressure_acceleration_jvp(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities, context.primal_scratch_.pressures, state_tangent.positions, parameter_tangent.particles, context.density_tangent_, context.tangent_scratch_.pressures, context.tangent_scratch_.pressure_accelerations);
            sph::pressure_acceleration_forward(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities, context.primal_scratch_.pressures, context.primal_scratch_.pressure_accelerations);
        }
        cuda_kernel::launch_predict_jvp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, collision_domain(configuration.fluid, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(step_cache.non_pressure_accelerations), kernel_vector(context.primal_scratch_.pressure_accelerations), kernel_vector(state_tangent.positions), kernel_vector(state_tangent.velocities), kernel_vector(context.non_pressure_acceleration_tangent_), kernel_vector(context.tangent_scratch_.pressure_accelerations), kernel_vector(next_state_tangent.positions), kernel_vector(next_state_tangent.velocities));
    }

    void Model::vjp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const {
        sph::zero_scalar_adjoint(*context.resource, context.density_adjoint_);
        sph::zero_vector_adjoint(*context.resource, context.non_pressure_acceleration_adjoint_);
        zero_iteration(*context.resource, context.adjoint_scratch_);
        zero_iteration(*context.resource, context.previous_adjoint_scratch_);
        cuda_kernel::launch_predict_vjp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, collision_domain(configuration.fluid, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(step_cache.non_pressure_accelerations), kernel_vector(step_cache.checkpoints.back().pressure_accelerations), kernel_vector(next_state_adjoint.positions), kernel_vector(next_state_adjoint.velocities), kernel_vector(previous_state_adjoint.positions), kernel_vector(previous_state_adjoint.velocities), kernel_vector(context.non_pressure_acceleration_adjoint_), kernel_vector(context.adjoint_scratch_.pressure_accelerations));

        for (std::size_t checkpoint = step_cache.checkpoints.size() - 1u; checkpoint > 0u; --checkpoint) {
            const IterationCache& first = step_cache.checkpoints[checkpoint - 1u];
            const IterationCache& last = step_cache.checkpoints[checkpoint];
            copy_iteration(*context.resource, first, context.recomputed_iterations_[0], configuration.fluid.particle_count);
            for (std::uint32_t iteration = first.iteration + 1u; iteration <= last.iteration; ++iteration) {
                IterationCache& previous = context.recomputed_iterations_[iteration - first.iteration - 1u];
                IterationCache& current = context.recomputed_iterations_[iteration - first.iteration];
                current.iteration = iteration;
                cuda_kernel::launch_predict_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, collision_domain(configuration.fluid, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(step_cache.non_pressure_accelerations), kernel_vector(previous.pressure_accelerations), kernel_vector(current.predicted_positions), kernel_vector(current.predicted_velocities));
                sph::density_forward_frozen(*context.resource, configuration.fluid, context.boundary, state.positions, current.predicted_positions, parameters.particles, step_cache.neighborhood, current.predicted_densities);
                cuda_kernel::launch_jacobi_update_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, reference_gradient_norm_, kernel_parameters(parameters.particles), step_cache.densities.values.data, previous.pressures.values.data, current.predicted_densities.values.data, parameters.jacobi_relaxation.data, current.pressures.values.data);
                sph::pressure_acceleration_forward(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities, current.pressures, current.pressure_accelerations);
            }
            for (std::uint32_t iteration = last.iteration; iteration > first.iteration; --iteration) {
                IterationCache& previous = context.recomputed_iterations_[iteration - first.iteration - 1u];
                IterationCache& current = context.recomputed_iterations_[iteration - first.iteration];
                zero_iteration(*context.resource, context.previous_adjoint_scratch_);
                sph::pressure_acceleration_vjp(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities, current.pressures, context.adjoint_scratch_.pressure_accelerations, previous_state_adjoint.positions, context.density_adjoint_, context.adjoint_scratch_.pressures, parameter_adjoint.particles);
                cuda_kernel::launch_jacobi_update_vjp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, reference_gradient_norm_, kernel_parameters(parameters.particles), step_cache.densities.values.data, previous.pressures.values.data, current.predicted_densities.values.data, parameters.jacobi_relaxation.data, context.adjoint_scratch_.pressures.values.data, kernel_parameters(parameter_adjoint.particles), context.density_adjoint_.values.data, context.previous_adjoint_scratch_.pressures.values.data, context.adjoint_scratch_.predicted_densities.values.data, parameter_adjoint.jacobi_relaxation.data);
                sph::density_vjp_frozen(*context.resource, configuration.fluid, context.boundary, state.positions, current.predicted_positions, parameters.particles, step_cache.neighborhood, context.adjoint_scratch_.predicted_densities, context.adjoint_scratch_.predicted_positions, parameter_adjoint.particles);
                cuda_kernel::launch_predict_vjp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.time_step, collision_domain(configuration.fluid, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(step_cache.non_pressure_accelerations), kernel_vector(previous.pressure_accelerations), kernel_vector(context.adjoint_scratch_.predicted_positions), kernel_vector(context.adjoint_scratch_.predicted_velocities), kernel_vector(previous_state_adjoint.positions), kernel_vector(previous_state_adjoint.velocities), kernel_vector(context.non_pressure_acceleration_adjoint_), kernel_vector(context.previous_adjoint_scratch_.pressure_accelerations));
                std::swap(context.adjoint_scratch_, context.previous_adjoint_scratch_);
            }
        }
        sph::non_pressure_vjp(*context.resource, configuration.fluid, context.boundary, state, parameters.particles, step_cache.neighborhood, step_cache.densities, context.non_pressure_acceleration_adjoint_, previous_state_adjoint, control_adjoint, context.density_adjoint_, parameter_adjoint.particles);
        sph::density_vjp(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, context.density_adjoint_, previous_state_adjoint.positions, parameter_adjoint.particles);
    }

} // namespace xayah::fluid::iisph
