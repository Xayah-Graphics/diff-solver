module xayah.fluid.flip;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.grid;

namespace xayah::fluid::flip {

    Model::Model(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ExecutionContext Model::allocate_context(const fluid::ExecutionMode mode) const {
        ExecutionContext context{};
        context.resource = std::make_shared<cuda::Resource>();
        context.domain   = grid::allocate_domain(context.resource, configuration.grid);
        if (mode == fluid::ExecutionMode::differentiable) {
            context.grid_differential_scratch_      = grid::allocate_differential_scratch(context.resource, configuration.grid);
            context.transferred_velocity_tangent_   = grid::allocate_vector_field(context.resource, configuration.particle_count);
            context.transferred_velocity_adjoint_   = grid::allocate_vector_adjoint_field(context.resource, configuration.particle_count);
        }
        return context;
    }

    fluid::State Model::allocate_state(ExecutionContext& context) const {
        fluid::State value{
            .positions  = grid::allocate_vector_field(context.resource, configuration.particle_count),
            .velocities = grid::allocate_vector_field(context.resource, configuration.particle_count),
            .step_index = 0u,
        };
        grid::zero(*context.resource, value.positions);
        grid::zero(*context.resource, value.velocities);
        return value;
    }

    fluid::Control Model::allocate_control(ExecutionContext& context) const {
        fluid::Control value{.external_accelerations = grid::allocate_vector_field(context.resource, configuration.particle_count)};
        grid::zero(*context.resource, value.external_accelerations);
        return value;
    }

    Parameters Model::allocate_parameters(ExecutionContext& context) const {
        Parameters value{
            .masses   = cuda::Buffer<float>(context.resource, configuration.particle_count),
            .blending = cuda::Buffer<float>(context.resource, 1u),
        };
        context.resource->zero(value.masses.data, value.masses.size * sizeof(float));
        const float pure_flip = 1.0F;
        context.resource->copy_from_host(value.blending.data, &pure_flip, sizeof(float));
        return value;
    }

    StepCache Model::allocate_step_cache(ExecutionContext& context) const {
        return {
            .grid = grid::allocate_step_cache(context.resource, configuration.grid, configuration.particle_count),
            .transferred_velocities = grid::allocate_vector_field(context.resource, configuration.particle_count),
        };
    }

    fluid::StateTangent Model::allocate_state_tangent(ExecutionContext& context) const {
        fluid::StateTangent value{
            .positions  = grid::allocate_vector_field(context.resource, configuration.particle_count),
            .velocities = grid::allocate_vector_field(context.resource, configuration.particle_count),
        };
        grid::zero(*context.resource, value.positions);
        grid::zero(*context.resource, value.velocities);
        return value;
    }

    fluid::ControlTangent Model::allocate_control_tangent(ExecutionContext& context) const {
        fluid::ControlTangent value{.external_accelerations = grid::allocate_vector_field(context.resource, configuration.particle_count)};
        grid::zero(*context.resource, value.external_accelerations);
        return value;
    }

    ParameterTangent Model::allocate_parameter_tangent(ExecutionContext& context) const {
        ParameterTangent value{
            .masses   = cuda::Buffer<float>(context.resource, configuration.particle_count),
            .blending = cuda::Buffer<float>(context.resource, 1u),
        };
        context.resource->zero(value.masses.data, value.masses.size * sizeof(float));
        context.resource->zero(value.blending.data, sizeof(float));
        return value;
    }

    fluid::StateAdjoint Model::allocate_state_adjoint(ExecutionContext& context) const {
        fluid::StateAdjoint value{
            .positions  = grid::allocate_vector_adjoint_field(context.resource, configuration.particle_count),
            .velocities = grid::allocate_vector_adjoint_field(context.resource, configuration.particle_count),
        };
        grid::zero(*context.resource, value.positions);
        grid::zero(*context.resource, value.velocities);
        return value;
    }

    fluid::ControlAdjoint Model::allocate_control_adjoint(ExecutionContext& context) const {
        fluid::ControlAdjoint value{.external_accelerations = grid::allocate_vector_adjoint_field(context.resource, configuration.particle_count)};
        grid::zero(*context.resource, value.external_accelerations);
        return value;
    }

    ParameterAdjoint Model::allocate_parameter_adjoint(ExecutionContext& context) const {
        ParameterAdjoint value{
            .masses   = cuda::Buffer<double>(context.resource, configuration.particle_count),
            .blending = cuda::Buffer<double>(context.resource, 1u),
        };
        context.resource->zero(value.masses.data, value.masses.size * sizeof(double));
        context.resource->zero(value.blending.data, sizeof(double));
        return value;
    }

    void Model::copy_state(const fluid::State& source, fluid::State& destination, ExecutionContext& context) const {
        grid::copy(*context.resource, source.positions, destination.positions);
        grid::copy(*context.resource, source.velocities, destination.velocities);
        destination.step_index = source.step_index;
    }

    void Model::copy_state_tangent(const fluid::StateTangent& source, fluid::StateTangent& destination, ExecutionContext& context) const {
        grid::copy(*context.resource, source.positions, destination.positions);
        grid::copy(*context.resource, source.velocities, destination.velocities);
    }

    void Model::copy_state_adjoint(const fluid::StateAdjoint& source, fluid::StateAdjoint& destination, ExecutionContext& context) const {
        grid::copy(*context.resource, source.positions, destination.positions);
        grid::copy(*context.resource, source.velocities, destination.velocities);
    }

    void Model::accumulate_state_adjoint(const fluid::StateAdjoint& source, fluid::StateAdjoint& destination, ExecutionContext& context) const {
        grid::accumulate(*context.resource, source.positions, destination.positions);
        grid::accumulate(*context.resource, source.velocities, destination.velocities);
    }

    void Model::forward_step(const fluid::State& state, const fluid::Control& control, const Parameters& parameters, fluid::State& next_state, StepCache& step_cache, ExecutionContext& context) const {
        particle_to_grid_.forward(*context.resource, configuration.grid, grid::TransferKind::pic_flip, state.positions, state.velocities, nullptr, control.external_accelerations, parameters.masses, configuration.gravity, step_cache.grid.transfer);
        marker_.forward(*context.resource, configuration.grid, context.domain, step_cache.grid.transfer, step_cache.grid.projection);
        projection_.forward(*context.resource, configuration.grid, context.domain, step_cache.grid.projection, step_cache.grid.transfer.forced_velocity);
        extrapolation_.forward(*context.resource, configuration.grid, context.domain, step_cache.grid.projection, step_cache.grid.extrapolation);
        grid_to_particle_.forward_pic_flip(*context.resource, configuration.grid, state.positions, state.velocities, step_cache.grid.transfer, step_cache.grid.extrapolation, parameters.blending, step_cache.transferred_velocities);
        advection_.forward(*context.resource, configuration.grid, context.domain, configuration.particle_radius, state.positions, step_cache.transferred_velocities, step_cache.grid.advection, next_state.positions, next_state.velocities);
        next_state.step_index = state.step_index + 1u;
    }

    void Model::jvp_step(const fluid::State& state, const fluid::Control& control, const Parameters& parameters, const fluid::State&, const StepCache& step_cache, const fluid::StateTangent& state_tangent, const fluid::ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, fluid::StateTangent& next_state_tangent, ExecutionContext& context) const {
        grid::DifferentialScratch& scratch = context.grid_differential_scratch_;
        particle_to_grid_.jvp(*context.resource, configuration.grid, grid::TransferKind::pic_flip, state.positions, state.velocities, nullptr, control.external_accelerations, parameters.masses, configuration.gravity, state_tangent.positions, state_tangent.velocities, nullptr, control_tangent.external_accelerations, parameter_tangent.masses, step_cache.grid.transfer, scratch);
        projection_.jvp(*context.resource, configuration.grid, context.domain, step_cache.grid.projection, scratch.forced_velocity_tangent, scratch);
        extrapolation_.jvp(*context.resource, configuration.grid, context.domain, step_cache.grid.projection, step_cache.grid.extrapolation, scratch.projected_velocity_tangent, scratch);
        grid_to_particle_.jvp_pic_flip(*context.resource, configuration.grid, state.positions, state.velocities, step_cache.grid.transfer, step_cache.grid.extrapolation, parameters.blending, state_tangent.positions, state_tangent.velocities, parameter_tangent.blending, scratch, context.transferred_velocity_tangent_);
        advection_.jvp(*context.resource, configuration.grid, configuration.particle_radius, step_cache.grid.advection, state_tangent.positions, context.transferred_velocity_tangent_, next_state_tangent.positions, next_state_tangent.velocities);
    }

    void Model::vjp_step(const fluid::State& state, const fluid::Control& control, const Parameters& parameters, const fluid::State&, const StepCache& step_cache, const fluid::StateAdjoint& next_state_adjoint, fluid::StateAdjoint& previous_state_adjoint, fluid::ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const {
        grid::DifferentialScratch& scratch = context.grid_differential_scratch_;
        context.resource->zero(scratch.old_velocity_adjoint.x.data, scratch.old_velocity_adjoint.x.size * sizeof(double));
        context.resource->zero(scratch.old_velocity_adjoint.y.data, scratch.old_velocity_adjoint.y.size * sizeof(double));
        context.resource->zero(scratch.old_velocity_adjoint.z.data, scratch.old_velocity_adjoint.z.size * sizeof(double));
        context.resource->zero(scratch.forced_velocity_adjoint.x.data, scratch.forced_velocity_adjoint.x.size * sizeof(double));
        context.resource->zero(scratch.forced_velocity_adjoint.y.data, scratch.forced_velocity_adjoint.y.size * sizeof(double));
        context.resource->zero(scratch.forced_velocity_adjoint.z.data, scratch.forced_velocity_adjoint.z.size * sizeof(double));
        context.resource->zero(scratch.divergence_adjoint.data, scratch.divergence_adjoint.size * sizeof(double));
        context.resource->zero(scratch.pressure_adjoint_history.data, scratch.pressure_adjoint_history.size * sizeof(double));
        context.resource->zero(scratch.projected_velocity_adjoint.x.data, scratch.projected_velocity_adjoint.x.size * sizeof(double));
        context.resource->zero(scratch.projected_velocity_adjoint.y.data, scratch.projected_velocity_adjoint.y.size * sizeof(double));
        context.resource->zero(scratch.projected_velocity_adjoint.z.data, scratch.projected_velocity_adjoint.z.size * sizeof(double));
        context.resource->zero(scratch.extrapolated_velocity_adjoint_history.data, scratch.extrapolated_velocity_adjoint_history.size * sizeof(double));
        context.resource->zero(scratch.extrapolated_velocity_adjoint.x.data, scratch.extrapolated_velocity_adjoint.x.size * sizeof(double));
        context.resource->zero(scratch.extrapolated_velocity_adjoint.y.data, scratch.extrapolated_velocity_adjoint.y.size * sizeof(double));
        context.resource->zero(scratch.extrapolated_velocity_adjoint.z.data, scratch.extrapolated_velocity_adjoint.z.size * sizeof(double));
        grid::zero(*context.resource, context.transferred_velocity_adjoint_);

        advection_.vjp(*context.resource, configuration.grid, configuration.particle_radius, step_cache.grid.advection, next_state_adjoint.positions, next_state_adjoint.velocities, previous_state_adjoint.positions, context.transferred_velocity_adjoint_);
        grid_to_particle_.vjp_pic_flip(*context.resource, configuration.grid, state.positions, state.velocities, step_cache.grid.transfer, step_cache.grid.extrapolation, parameters.blending, context.transferred_velocity_adjoint_, previous_state_adjoint.positions, previous_state_adjoint.velocities, scratch.old_velocity_adjoint, scratch.extrapolated_velocity_adjoint, parameter_adjoint.blending);
        extrapolation_.vjp(*context.resource, configuration.grid, context.domain, step_cache.grid.projection, step_cache.grid.extrapolation, scratch, scratch.projected_velocity_adjoint);
        projection_.vjp(*context.resource, configuration.grid, context.domain, step_cache.grid.projection, scratch, scratch.forced_velocity_adjoint);
        particle_to_grid_.vjp(*context.resource, configuration.grid, grid::TransferKind::pic_flip, state.positions, state.velocities, nullptr, control.external_accelerations, parameters.masses, configuration.gravity, step_cache.grid.transfer, scratch, previous_state_adjoint.positions, previous_state_adjoint.velocities, nullptr, control_adjoint.external_accelerations, parameter_adjoint.masses);
    }

} // namespace xayah::fluid::flip
