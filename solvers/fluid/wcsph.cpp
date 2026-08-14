module;

#include "sph.h"
#include "wcsph.h"

module xayah.fluid.wcsph;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.sph;

namespace xayah::fluid::wcsph {

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

        sph::cuda_kernel::Neighborhood kernel_neighborhood(const Neighborhood& neighborhood) {
            return {
                .sorted_keys             = neighborhood.sorted_keys.data,
                .sorted_particle_indices = neighborhood.sorted_particle_indices.data,
                .cell_offsets             = neighborhood.cell_offsets.data,
                .sorted_boundary_keys     = neighborhood.sorted_boundary_keys.data,
                .sorted_boundary_indices  = neighborhood.sorted_boundary_indices.data,
                .boundary_cell_offsets    = neighborhood.boundary_cell_offsets.data,
                .particle_count           = static_cast<std::uint32_t>(neighborhood.sorted_keys.size),
                .boundary_count           = static_cast<std::uint32_t>(neighborhood.sorted_boundary_keys.size),
                .cells_x                  = neighborhood.cell_resolution[0],
                .cells_y                  = neighborhood.cell_resolution[1],
                .cells_z                  = neighborhood.cell_resolution[2],
                .origin_x                 = neighborhood.cell_origin.x,
                .origin_y                 = neighborhood.cell_origin.y,
                .origin_z                 = neighborhood.cell_origin.z,
                .cell_size                = neighborhood.cell_size,
            };
        }

        sph::cuda_kernel::Boundary kernel_boundary(const sph::DeviceBoundary& boundary, const Neighborhood& neighborhood) {
            return {
                .position_x = boundary.positions.x.data,
                .position_y = boundary.positions.y.data,
                .position_z = boundary.positions.z.data,
                .velocity_x = boundary.velocities.x.data,
                .velocity_y = boundary.velocities.y.data,
                .velocity_z = boundary.velocities.z.data,
                .volumes    = boundary.volumes.values.data,
                .count      = static_cast<std::uint32_t>(boundary.volumes.values.size),
                .time       = neighborhood.boundary_time,
            };
        }

        void zero_parameters(cuda::Resource& resource, ParameterTangent& parameters) {
            resource.zero(parameters.particles.masses.data, parameters.particles.masses.size * sizeof(float));
            resource.zero(parameters.particles.rest_densities.data, parameters.particles.rest_densities.size * sizeof(float));
            resource.zero(parameters.particles.viscosities.data, parameters.particles.viscosities.size * sizeof(float));
            resource.zero(parameters.particles.surface_tensions.data, parameters.particles.surface_tensions.size * sizeof(float));
            resource.zero(parameters.speed_of_sound.data, parameters.speed_of_sound.size * sizeof(float));
            resource.zero(parameters.tait_exponent.data, parameters.tait_exponent.size * sizeof(float));
            resource.zero(parameters.boundary_surface_tension.data, parameters.boundary_surface_tension.size * sizeof(float));
        }

        void zero_parameters(cuda::Resource& resource, ParameterAdjoint& parameters) {
            resource.zero(parameters.particles.masses.data, parameters.particles.masses.size * sizeof(double));
            resource.zero(parameters.particles.rest_densities.data, parameters.particles.rest_densities.size * sizeof(double));
            resource.zero(parameters.particles.viscosities.data, parameters.particles.viscosities.size * sizeof(double));
            resource.zero(parameters.particles.surface_tensions.data, parameters.particles.surface_tensions.size * sizeof(double));
            resource.zero(parameters.speed_of_sound.data, parameters.speed_of_sound.size * sizeof(double));
            resource.zero(parameters.tait_exponent.data, parameters.tait_exponent.size * sizeof(double));
            resource.zero(parameters.boundary_surface_tension.data, parameters.boundary_surface_tension.size * sizeof(double));
        }

    } // namespace

    Model::Model(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ExecutionContext Model::allocate_context(const ExecutionMode mode) const {
        ExecutionContext context{};
        context.resource = std::make_shared<cuda::Resource>();
        context.neighbor_search = sph::allocate_neighbor_search(context.resource, configuration.fluid);
        context.boundary = sph::allocate_boundary(context.resource, configuration.fluid);
        if (mode == ExecutionMode::differentiable) {
            context.density_tangent_ = sph::allocate_scalar_field(context.resource, configuration.fluid.particle_count);
            context.pressure_tangent_ = sph::allocate_scalar_field(context.resource, configuration.fluid.particle_count);
            context.pressure_acceleration_tangent_ = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count);
            context.viscosity_acceleration_tangent_ = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count);
            context.surface_acceleration_tangent_ = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count);
            context.total_acceleration_tangent_ = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count);
            context.density_adjoint_ = sph::allocate_scalar_adjoint_field(context.resource, configuration.fluid.particle_count);
            context.pressure_adjoint_ = sph::allocate_scalar_adjoint_field(context.resource, configuration.fluid.particle_count);
            context.pressure_acceleration_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, configuration.fluid.particle_count);
            context.viscosity_acceleration_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, configuration.fluid.particle_count);
            context.surface_acceleration_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, configuration.fluid.particle_count);
            context.total_acceleration_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, configuration.fluid.particle_count);
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
            .particles      = sph::allocate_particle_parameters(context.resource, configuration.fluid.particle_count),
            .speed_of_sound = cuda::Buffer<float>(context.resource, configuration.fluid.particle_count),
            .tait_exponent  = cuda::Buffer<float>(context.resource, configuration.fluid.particle_count),
            .boundary_surface_tension = cuda::Buffer<float>(context.resource, configuration.fluid.particle_count),
        };
    }

    StepCache Model::allocate_step_cache(ExecutionContext& context) const {
        return {
            .neighborhood            = sph::allocate_neighborhood(context.resource, configuration.fluid),
            .densities               = sph::allocate_scalar_field(context.resource, configuration.fluid.particle_count),
            .pressures               = sph::allocate_scalar_field(context.resource, configuration.fluid.particle_count),
            .pressure_accelerations  = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count),
            .viscosity_accelerations = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count),
            .surface_accelerations   = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count),
            .external_accelerations  = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count),
            .total_accelerations     = sph::allocate_vector_field(context.resource, configuration.fluid.particle_count),
        };
    }

    StateTangent Model::allocate_state_tangent(ExecutionContext& context) const {
        return sph::allocate_state_tangent(context.resource, configuration.fluid.particle_count);
    }

    ControlTangent Model::allocate_control_tangent(ExecutionContext& context) const {
        return sph::allocate_control_tangent(context.resource, configuration.fluid.particle_count);
    }

    ParameterTangent Model::allocate_parameter_tangent(ExecutionContext& context) const {
        ParameterTangent tangent{
            .particles      = sph::allocate_particle_parameter_tangent(context.resource, configuration.fluid.particle_count),
            .speed_of_sound = cuda::Buffer<float>(context.resource, configuration.fluid.particle_count),
            .tait_exponent  = cuda::Buffer<float>(context.resource, configuration.fluid.particle_count),
            .boundary_surface_tension = cuda::Buffer<float>(context.resource, configuration.fluid.particle_count),
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
            .particles      = sph::allocate_particle_parameter_adjoint(context.resource, configuration.fluid.particle_count),
            .speed_of_sound = cuda::Buffer<double>(context.resource, configuration.fluid.particle_count),
            .tait_exponent  = cuda::Buffer<double>(context.resource, configuration.fluid.particle_count),
            .boundary_surface_tension = cuda::Buffer<double>(context.resource, configuration.fluid.particle_count),
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
        cuda_kernel::launch_eos_forward(context.resource->native_stream, configuration.fluid.particle_count, step_cache.densities.values.data, kernel_parameters(parameters.particles), parameters.speed_of_sound.data, parameters.tait_exponent.data, step_cache.pressures.values.data);
        sph::pressure_acceleration_forward(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities, step_cache.pressures, step_cache.pressure_accelerations);
        cuda_kernel::launch_artificial_viscosity_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.support_radius, kernel_vector(state.positions), kernel_vector(state.velocities), kernel_parameters(parameters.particles), parameters.speed_of_sound.data, kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), step_cache.densities.values.data, kernel_vector(step_cache.viscosity_accelerations));
        cuda_kernel::launch_surface_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.support_radius, configuration.fluid.particle_radius, kernel_vector(state.positions), kernel_parameters(parameters.particles), parameters.boundary_surface_tension.data, kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), kernel_vector(step_cache.surface_accelerations));
        sph::add_accelerations(*context.resource, step_cache.pressure_accelerations, step_cache.viscosity_accelerations, step_cache.total_accelerations);
        sph::add_accelerations(*context.resource, step_cache.total_accelerations, step_cache.surface_accelerations, step_cache.total_accelerations);
        cuda_kernel::launch_external_forward(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.gravity.x, configuration.fluid.gravity.y, configuration.fluid.gravity.z, kernel_vector(control.external_accelerations), kernel_vector(step_cache.external_accelerations));
        sph::add_accelerations(*context.resource, step_cache.total_accelerations, step_cache.external_accelerations, step_cache.total_accelerations);
        sph::integrate_forward(*context.resource, configuration.fluid, state, step_cache.total_accelerations, next_state);
    }

    void Model::jvp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const {
        sph::density_jvp(*context.resource, configuration.fluid, context.boundary, state.positions, state_tangent.positions, parameters.particles, parameter_tangent.particles, step_cache.neighborhood, context.density_tangent_);
        cuda_kernel::launch_eos_jvp(context.resource->native_stream, configuration.fluid.particle_count, step_cache.densities.values.data, context.density_tangent_.values.data, kernel_parameters(parameters.particles), kernel_parameters(parameter_tangent.particles), parameters.speed_of_sound.data, parameter_tangent.speed_of_sound.data, parameters.tait_exponent.data, parameter_tangent.tait_exponent.data, context.pressure_tangent_.values.data);
        sph::pressure_acceleration_jvp(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities, step_cache.pressures, state_tangent.positions, parameter_tangent.particles, context.density_tangent_, context.pressure_tangent_, context.pressure_acceleration_tangent_);
        cuda_kernel::launch_artificial_viscosity_jvp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.support_radius, kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(state_tangent.positions), kernel_vector(state_tangent.velocities), kernel_parameters(parameters.particles), kernel_parameters(parameter_tangent.particles), parameters.speed_of_sound.data, parameter_tangent.speed_of_sound.data, kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), step_cache.densities.values.data, context.density_tangent_.values.data, kernel_vector(context.viscosity_acceleration_tangent_));
        cuda_kernel::launch_surface_jvp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.support_radius, configuration.fluid.particle_radius, kernel_vector(state.positions), kernel_vector(state_tangent.positions), kernel_parameters(parameters.particles), kernel_parameters(parameter_tangent.particles), parameters.boundary_surface_tension.data, parameter_tangent.boundary_surface_tension.data, kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), kernel_vector(context.surface_acceleration_tangent_));
        sph::add_accelerations(*context.resource, context.pressure_acceleration_tangent_, context.viscosity_acceleration_tangent_, context.total_acceleration_tangent_);
        sph::add_accelerations(*context.resource, context.total_acceleration_tangent_, context.surface_acceleration_tangent_, context.total_acceleration_tangent_);
        sph::add_accelerations(*context.resource, context.total_acceleration_tangent_, control_tangent.external_accelerations, context.total_acceleration_tangent_);
        sph::integrate_jvp(*context.resource, configuration.fluid, state, step_cache.total_accelerations, state_tangent, context.total_acceleration_tangent_, next_state_tangent);
    }

    void Model::vjp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const {
        sph::zero_scalar_adjoint(*context.resource, context.density_adjoint_);
        sph::zero_scalar_adjoint(*context.resource, context.pressure_adjoint_);
        sph::zero_vector_adjoint(*context.resource, context.total_acceleration_adjoint_);
        sph::integrate_vjp(*context.resource, configuration.fluid, state, step_cache.total_accelerations, next_state_adjoint, previous_state_adjoint, context.total_acceleration_adjoint_);
        sph::accumulate_vector_adjoint(*context.resource, context.total_acceleration_adjoint_, control_adjoint.external_accelerations);
        cuda_kernel::launch_surface_vjp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.support_radius, configuration.fluid.particle_radius, kernel_vector(state.positions), kernel_parameters(parameters.particles), parameters.boundary_surface_tension.data, kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), kernel_vector(context.total_acceleration_adjoint_), kernel_vector(previous_state_adjoint.positions), kernel_parameters(parameter_adjoint.particles), parameter_adjoint.boundary_surface_tension.data);
        cuda_kernel::launch_artificial_viscosity_vjp(context.resource->native_stream, configuration.fluid.particle_count, configuration.fluid.support_radius, kernel_vector(state.positions), kernel_vector(state.velocities), kernel_parameters(parameters.particles), parameters.speed_of_sound.data, kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), step_cache.densities.values.data, kernel_vector(context.total_acceleration_adjoint_), kernel_vector(previous_state_adjoint.positions), kernel_vector(previous_state_adjoint.velocities), context.density_adjoint_.values.data, kernel_parameters(parameter_adjoint.particles), parameter_adjoint.speed_of_sound.data);
        sph::pressure_acceleration_vjp(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, step_cache.densities, step_cache.pressures, context.total_acceleration_adjoint_, previous_state_adjoint.positions, context.density_adjoint_, context.pressure_adjoint_, parameter_adjoint.particles);
        cuda_kernel::launch_eos_vjp(context.resource->native_stream, configuration.fluid.particle_count, step_cache.densities.values.data, kernel_parameters(parameters.particles), parameters.speed_of_sound.data, parameters.tait_exponent.data, context.pressure_adjoint_.values.data, context.density_adjoint_.values.data, kernel_parameters(parameter_adjoint.particles), parameter_adjoint.speed_of_sound.data, parameter_adjoint.tait_exponent.data);
        sph::density_vjp(*context.resource, configuration.fluid, context.boundary, state.positions, parameters.particles, step_cache.neighborhood, context.density_adjoint_, previous_state_adjoint.positions, parameter_adjoint.particles);
    }

} // namespace xayah::fluid::wcsph
