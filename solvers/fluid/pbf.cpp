module;

#include "pbf.h"

module xayah.fluid.pbf;

import std;
import xayah.cuda;
import xayah.fluid.data;
import xayah.fluid.sph;

namespace xayah::fluid::pbf {

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
                .sorted_keys = neighborhood.sorted_keys.data,
                .sorted_particle_indices = neighborhood.sorted_particle_indices.data,
                .cell_offsets = neighborhood.cell_offsets.data,
                .sorted_boundary_keys = neighborhood.sorted_boundary_keys.data,
                .sorted_boundary_indices = neighborhood.sorted_boundary_indices.data,
                .boundary_cell_offsets = neighborhood.boundary_cell_offsets.data,
                .particle_count = static_cast<std::uint32_t>(neighborhood.sorted_keys.size),
                .boundary_count = static_cast<std::uint32_t>(neighborhood.sorted_boundary_keys.size),
                .cells_x = neighborhood.cell_resolution[0],
                .cells_y = neighborhood.cell_resolution[1],
                .cells_z = neighborhood.cell_resolution[2],
                .origin_x = neighborhood.cell_origin.x,
                .origin_y = neighborhood.cell_origin.y,
                .origin_z = neighborhood.cell_origin.z,
                .cell_size = neighborhood.cell_size,
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
                .volumes = boundary.volumes.values.data,
                .count = static_cast<std::uint32_t>(boundary.volumes.values.size),
                .time = neighborhood.boundary_time,
            };
        }

        sph::cuda_kernel::Domain kernel_domain(const Configuration& configuration, const std::uint64_t step_index) {
            const float time = static_cast<float>(step_index) * configuration.fluid.time_step;
            return {
                .minimum_x = configuration.fluid.domain.minimum.x + time * configuration.fluid.domain.velocity.x + configuration.fluid.particle_radius,
                .minimum_y = configuration.fluid.domain.minimum.y + time * configuration.fluid.domain.velocity.y + configuration.fluid.particle_radius,
                .minimum_z = configuration.fluid.domain.minimum.z + time * configuration.fluid.domain.velocity.z + configuration.fluid.particle_radius,
                .maximum_x = configuration.fluid.domain.maximum.x + time * configuration.fluid.domain.velocity.x - configuration.fluid.particle_radius,
                .maximum_y = configuration.fluid.domain.maximum.y + time * configuration.fluid.domain.velocity.y - configuration.fluid.particle_radius,
                .maximum_z = configuration.fluid.domain.maximum.z + time * configuration.fluid.domain.velocity.z - configuration.fluid.particle_radius,
                .velocity_x = configuration.fluid.domain.velocity.x,
                .velocity_y = configuration.fluid.domain.velocity.y,
                .velocity_z = configuration.fluid.domain.velocity.z,
                .no_slip = configuration.fluid.domain.no_slip ? 1u : 0u,
            };
        }

        void zero_parameters(cuda::Resource& resource, ParameterTangent& parameters) {
            resource.zero(parameters.particles.masses.data, parameters.particles.masses.size * sizeof(float));
            resource.zero(parameters.particles.rest_densities.data, parameters.particles.rest_densities.size * sizeof(float));
            resource.zero(parameters.particles.viscosities.data, parameters.particles.viscosities.size * sizeof(float));
            resource.zero(parameters.particles.surface_tensions.data, parameters.particles.surface_tensions.size * sizeof(float));
            resource.zero(parameters.relaxation.data, parameters.relaxation.size * sizeof(float));
            resource.zero(parameters.artificial_pressure_strength.data, parameters.artificial_pressure_strength.size * sizeof(float));
            resource.zero(parameters.artificial_pressure_exponent.data, parameters.artificial_pressure_exponent.size * sizeof(float));
            resource.zero(parameters.artificial_pressure_radius.data, parameters.artificial_pressure_radius.size * sizeof(float));
            resource.zero(parameters.xsph_viscosity.data, parameters.xsph_viscosity.size * sizeof(float));
            resource.zero(parameters.vorticity_confinement.data, parameters.vorticity_confinement.size * sizeof(float));
        }

        void zero_parameters(cuda::Resource& resource, ParameterAdjoint& parameters) {
            resource.zero(parameters.particles.masses.data, parameters.particles.masses.size * sizeof(double));
            resource.zero(parameters.particles.rest_densities.data, parameters.particles.rest_densities.size * sizeof(double));
            resource.zero(parameters.particles.viscosities.data, parameters.particles.viscosities.size * sizeof(double));
            resource.zero(parameters.particles.surface_tensions.data, parameters.particles.surface_tensions.size * sizeof(double));
            resource.zero(parameters.relaxation.data, parameters.relaxation.size * sizeof(double));
            resource.zero(parameters.artificial_pressure_strength.data, parameters.artificial_pressure_strength.size * sizeof(double));
            resource.zero(parameters.artificial_pressure_exponent.data, parameters.artificial_pressure_exponent.size * sizeof(double));
            resource.zero(parameters.artificial_pressure_radius.data, parameters.artificial_pressure_radius.size * sizeof(double));
            resource.zero(parameters.xsph_viscosity.data, parameters.xsph_viscosity.size * sizeof(double));
            resource.zero(parameters.vorticity_confinement.data, parameters.vorticity_confinement.size * sizeof(double));
        }

    } // namespace

    Model::Model(Configuration next_configuration) : configuration(std::move(next_configuration)) {}

    ExecutionContext Model::allocate_context(const ExecutionMode mode) const {
        ExecutionContext context{};
        context.resource = std::make_shared<cuda::Resource>();
        context.neighbor_search = sph::allocate_neighbor_search(context.resource, configuration.fluid);
        context.boundary = sph::allocate_boundary(context.resource, configuration.fluid);
        const std::size_t count = configuration.fluid.particle_count;
        context.forward_iteration_ = {
            .densities = sph::allocate_scalar_field(context.resource, count),
            .gradient_sums = sph::allocate_vector_field(context.resource, count),
            .denominators = sph::allocate_scalar_field(context.resource, count),
            .lambdas = sph::allocate_scalar_field(context.resource, count),
            .corrections = sph::allocate_vector_field(context.resource, count),
            .collision_masks = cuda::Buffer<std::uint32_t>(context.resource, count),
        };
        if (mode == ExecutionMode::differentiable) {
            context.segment_position_history_.reserve(configuration.checkpoint_interval + 1u);
            for (std::uint32_t index = 0u; index <= configuration.checkpoint_interval; ++index) context.segment_position_history_.push_back(sph::allocate_vector_field(context.resource, count));
            context.segment_iteration_history_.reserve(configuration.checkpoint_interval);
            for (std::uint32_t index = 0u; index < configuration.checkpoint_interval; ++index) {
                context.segment_iteration_history_.push_back({
                    .densities = sph::allocate_scalar_field(context.resource, count),
                    .gradient_sums = sph::allocate_vector_field(context.resource, count),
                    .denominators = sph::allocate_scalar_field(context.resource, count),
                    .lambdas = sph::allocate_scalar_field(context.resource, count),
                    .corrections = sph::allocate_vector_field(context.resource, count),
                    .collision_masks = cuda::Buffer<std::uint32_t>(context.resource, count),
                });
            }
            context.current_position_tangent_ = sph::allocate_vector_field(context.resource, count);
            context.next_position_tangent_ = sph::allocate_vector_field(context.resource, count);
            context.density_tangent_ = sph::allocate_scalar_field(context.resource, count);
            context.gradient_sum_tangent_ = sph::allocate_vector_field(context.resource, count);
            context.denominator_tangent_ = sph::allocate_scalar_field(context.resource, count);
            context.lambda_tangent_ = sph::allocate_scalar_field(context.resource, count);
            context.correction_tangent_ = sph::allocate_vector_field(context.resource, count);
            context.reconstructed_velocity_tangent_ = sph::allocate_vector_field(context.resource, count);
            context.vorticity_tangent_ = sph::allocate_vector_field(context.resource, count);
            context.vorticity_magnitude_tangent_ = sph::allocate_scalar_field(context.resource, count);
            context.vorticity_normal_tangent_ = sph::allocate_vector_field(context.resource, count);
            context.confined_velocity_tangent_ = sph::allocate_vector_field(context.resource, count);
            context.current_position_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, count);
            context.next_position_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, count);
            context.correction_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, count);
            context.lambda_adjoint_ = sph::allocate_scalar_adjoint_field(context.resource, count);
            context.density_adjoint_ = sph::allocate_scalar_adjoint_field(context.resource, count);
            context.reconstructed_velocity_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, count);
            context.vorticity_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, count);
            context.vorticity_magnitude_adjoint_ = sph::allocate_scalar_adjoint_field(context.resource, count);
            context.vorticity_normal_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, count);
            context.confined_velocity_adjoint_ = sph::allocate_vector_adjoint_field(context.resource, count);
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
        const std::size_t count = configuration.fluid.particle_count;
        return {
            .particles = sph::allocate_particle_parameters(context.resource, count),
            .relaxation = cuda::Buffer<float>(context.resource, count),
            .artificial_pressure_strength = cuda::Buffer<float>(context.resource, count),
            .artificial_pressure_exponent = cuda::Buffer<float>(context.resource, count),
            .artificial_pressure_radius = cuda::Buffer<float>(context.resource, count),
            .xsph_viscosity = cuda::Buffer<float>(context.resource, count),
            .vorticity_confinement = cuda::Buffer<float>(context.resource, count),
        };
    }

    StepCache Model::allocate_step_cache(ExecutionContext& context) const {
        const std::size_t count = configuration.fluid.particle_count;
        StepCache cache{
            .neighborhood = sph::allocate_neighborhood(context.resource, configuration.fluid),
            .predicted_positions = sph::allocate_vector_field(context.resource, count),
            .corrected_positions = sph::allocate_vector_field(context.resource, count),
            .checkpoints = {},
            .reconstructed_velocities = sph::allocate_vector_field(context.resource, count),
            .vorticities = sph::allocate_vector_field(context.resource, count),
            .vorticity_magnitudes = sph::allocate_scalar_field(context.resource, count),
            .vorticity_normals = sph::allocate_vector_field(context.resource, count),
            .vorticity_normalizers = sph::allocate_scalar_field(context.resource, count),
            .confined_velocities = sph::allocate_vector_field(context.resource, count),
        };
        cache.checkpoints.push_back({.iteration = 0u, .positions = sph::allocate_vector_field(context.resource, count)});
        for (std::uint32_t iteration = configuration.checkpoint_interval; iteration < configuration.pressure_iterations; iteration += configuration.checkpoint_interval) cache.checkpoints.push_back({.iteration = iteration, .positions = sph::allocate_vector_field(context.resource, count)});
        if (configuration.pressure_iterations != 0u) cache.checkpoints.push_back({.iteration = configuration.pressure_iterations, .positions = sph::allocate_vector_field(context.resource, count)});
        return cache;
    }

    StateTangent Model::allocate_state_tangent(ExecutionContext& context) const {
        return sph::allocate_state_tangent(context.resource, configuration.fluid.particle_count);
    }

    ControlTangent Model::allocate_control_tangent(ExecutionContext& context) const {
        return sph::allocate_control_tangent(context.resource, configuration.fluid.particle_count);
    }

    ParameterTangent Model::allocate_parameter_tangent(ExecutionContext& context) const {
        const std::size_t count = configuration.fluid.particle_count;
        ParameterTangent tangent{
            .particles = sph::allocate_particle_parameter_tangent(context.resource, count),
            .relaxation = cuda::Buffer<float>(context.resource, count),
            .artificial_pressure_strength = cuda::Buffer<float>(context.resource, count),
            .artificial_pressure_exponent = cuda::Buffer<float>(context.resource, count),
            .artificial_pressure_radius = cuda::Buffer<float>(context.resource, count),
            .xsph_viscosity = cuda::Buffer<float>(context.resource, count),
            .vorticity_confinement = cuda::Buffer<float>(context.resource, count),
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
        const std::size_t count = configuration.fluid.particle_count;
        ParameterAdjoint adjoint{
            .particles = sph::allocate_particle_parameter_adjoint(context.resource, count),
            .relaxation = cuda::Buffer<double>(context.resource, count),
            .artificial_pressure_strength = cuda::Buffer<double>(context.resource, count),
            .artificial_pressure_exponent = cuda::Buffer<double>(context.resource, count),
            .artificial_pressure_radius = cuda::Buffer<double>(context.resource, count),
            .xsph_viscosity = cuda::Buffer<double>(context.resource, count),
            .vorticity_confinement = cuda::Buffer<double>(context.resource, count),
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
        const std::uint32_t count = configuration.fluid.particle_count;
        cuda_kernel::launch_predict_forward(context.resource->native_stream, count, configuration.fluid.time_step, configuration.fluid.gravity.x, configuration.fluid.gravity.y, configuration.fluid.gravity.z, kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(control.external_accelerations), kernel_vector(step_cache.predicted_positions));
        sph::build_neighborhood(*context.resource, configuration.fluid, state.step_index + 1u, context.boundary, step_cache.predicted_positions, context.neighbor_search, step_cache.neighborhood);
        sph::copy_vector(*context.resource, step_cache.predicted_positions, step_cache.corrected_positions);
        sph::copy_vector(*context.resource, step_cache.predicted_positions, step_cache.checkpoints.front().positions);
        std::size_t checkpoint = 1u;
        for (std::uint32_t iteration_index = 0u; iteration_index < configuration.pressure_iterations; ++iteration_index) {
            sph::pbf_density_forward_frozen(*context.resource, configuration.fluid, context.boundary, step_cache.predicted_positions, step_cache.corrected_positions, parameters.particles, step_cache.neighborhood, context.forward_iteration_.densities);
            cuda_kernel::launch_lambda_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_parameters(parameters.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), context.forward_iteration_.densities.values.data, parameters.relaxation.data, kernel_vector(context.forward_iteration_.gradient_sums), context.forward_iteration_.denominators.values.data, context.forward_iteration_.lambdas.values.data);
            cuda_kernel::launch_correction_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_parameters(parameters.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), context.forward_iteration_.lambdas.values.data, parameters.artificial_pressure_strength.data, parameters.artificial_pressure_exponent.data, parameters.artificial_pressure_radius.data, kernel_vector(context.forward_iteration_.corrections));
            cuda_kernel::launch_project_forward(context.resource->native_stream, count, kernel_domain(configuration, state.step_index + 1u), kernel_vector(step_cache.corrected_positions), kernel_vector(context.forward_iteration_.corrections), context.forward_iteration_.collision_masks.data, kernel_vector(step_cache.corrected_positions));
            if (checkpoint < step_cache.checkpoints.size() && step_cache.checkpoints[checkpoint].iteration == iteration_index + 1u) {
                sph::copy_vector(*context.resource, step_cache.corrected_positions, step_cache.checkpoints[checkpoint].positions);
                ++checkpoint;
            }
        }
        cuda_kernel::launch_reconstruct_forward(context.resource->native_stream, count, 1.0F / configuration.fluid.time_step, kernel_vector(state.positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.reconstructed_velocities));
        cuda_kernel::launch_vorticity_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.reconstructed_velocities), kernel_neighborhood(step_cache.neighborhood), kernel_vector(step_cache.vorticities));
        cuda_kernel::launch_normal_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.vorticities), kernel_neighborhood(step_cache.neighborhood), step_cache.vorticity_magnitudes.values.data, kernel_vector(step_cache.vorticity_normals), step_cache.vorticity_normalizers.values.data);
        cuda_kernel::launch_confinement_forward(context.resource->native_stream, count, configuration.fluid.time_step, kernel_vector(step_cache.reconstructed_velocities), kernel_vector(step_cache.vorticities), kernel_vector(step_cache.vorticity_normals), parameters.vorticity_confinement.data, kernel_vector(step_cache.confined_velocities));
        cuda_kernel::launch_xsph_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.confined_velocities), kernel_neighborhood(step_cache.neighborhood), parameters.xsph_viscosity.data, kernel_vector(next_state.velocities));
        sph::copy_vector(*context.resource, step_cache.corrected_positions, next_state.positions);
        next_state.step_index = state.step_index + 1u;
    }

    void Model::jvp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const {
        const std::uint32_t count = configuration.fluid.particle_count;
        cuda_kernel::launch_predict_jvp(context.resource->native_stream, count, configuration.fluid.time_step, kernel_vector(state_tangent.positions), kernel_vector(state_tangent.velocities), kernel_vector(control_tangent.external_accelerations), kernel_vector(context.current_position_tangent_));
        sph::copy_vector(*context.resource, step_cache.predicted_positions, context.segment_position_history_.front());
        for (std::uint32_t iteration_index = 0u; iteration_index < configuration.pressure_iterations; ++iteration_index) {
            sph::pbf_density_forward_frozen(*context.resource, configuration.fluid, context.boundary, step_cache.predicted_positions, context.segment_position_history_.front(), parameters.particles, step_cache.neighborhood, context.forward_iteration_.densities);
            sph::pbf_density_jvp_frozen(*context.resource, configuration.fluid, context.boundary, step_cache.predicted_positions, context.segment_position_history_.front(), context.current_position_tangent_, parameters.particles, parameter_tangent.particles, step_cache.neighborhood, context.density_tangent_);
            cuda_kernel::launch_lambda_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(context.segment_position_history_.front()), kernel_parameters(parameters.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), context.forward_iteration_.densities.values.data, parameters.relaxation.data, kernel_vector(context.forward_iteration_.gradient_sums), context.forward_iteration_.denominators.values.data, context.forward_iteration_.lambdas.values.data);
            cuda_kernel::launch_lambda_jvp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(context.segment_position_history_.front()), kernel_vector(context.current_position_tangent_), kernel_parameters(parameters.particles), kernel_parameters(parameter_tangent.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), context.forward_iteration_.densities.values.data, context.density_tangent_.values.data, parameters.relaxation.data, parameter_tangent.relaxation.data, kernel_vector(context.gradient_sum_tangent_), context.denominator_tangent_.values.data, context.lambda_tangent_.values.data);
            cuda_kernel::launch_correction_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(context.segment_position_history_.front()), kernel_parameters(parameters.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), context.forward_iteration_.lambdas.values.data, parameters.artificial_pressure_strength.data, parameters.artificial_pressure_exponent.data, parameters.artificial_pressure_radius.data, kernel_vector(context.forward_iteration_.corrections));
            cuda_kernel::launch_correction_jvp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(context.segment_position_history_.front()), kernel_vector(context.current_position_tangent_), kernel_parameters(parameters.particles), kernel_parameters(parameter_tangent.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), context.forward_iteration_.lambdas.values.data, context.lambda_tangent_.values.data, parameters.artificial_pressure_strength.data, parameter_tangent.artificial_pressure_strength.data, parameters.artificial_pressure_exponent.data, parameter_tangent.artificial_pressure_exponent.data, parameters.artificial_pressure_radius.data, parameter_tangent.artificial_pressure_radius.data, kernel_vector(context.correction_tangent_));
            cuda_kernel::launch_project_forward(context.resource->native_stream, count, kernel_domain(configuration, state.step_index + 1u), kernel_vector(context.segment_position_history_.front()), kernel_vector(context.forward_iteration_.corrections), context.forward_iteration_.collision_masks.data, kernel_vector(context.segment_position_history_.front()));
            cuda_kernel::launch_project_jvp(context.resource->native_stream, count, context.forward_iteration_.collision_masks.data, kernel_vector(context.current_position_tangent_), kernel_vector(context.correction_tangent_), kernel_vector(context.next_position_tangent_));
            std::swap(context.current_position_tangent_, context.next_position_tangent_);
        }
        cuda_kernel::launch_reconstruct_jvp(context.resource->native_stream, count, 1.0F / configuration.fluid.time_step, kernel_vector(state_tangent.positions), kernel_vector(context.current_position_tangent_), kernel_vector(context.reconstructed_velocity_tangent_));
        cuda_kernel::launch_vorticity_jvp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.reconstructed_velocities), kernel_vector(context.current_position_tangent_), kernel_vector(context.reconstructed_velocity_tangent_), kernel_neighborhood(step_cache.neighborhood), kernel_vector(context.vorticity_tangent_));
        cuda_kernel::launch_normal_jvp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.vorticities), kernel_vector(context.current_position_tangent_), kernel_vector(context.vorticity_tangent_), kernel_neighborhood(step_cache.neighborhood), step_cache.vorticity_magnitudes.values.data, kernel_vector(step_cache.vorticity_normals), step_cache.vorticity_normalizers.values.data, context.vorticity_magnitude_tangent_.values.data, kernel_vector(context.vorticity_normal_tangent_));
        cuda_kernel::launch_confinement_jvp(context.resource->native_stream, count, configuration.fluid.time_step, kernel_vector(step_cache.vorticities), kernel_vector(step_cache.vorticity_normals), parameters.vorticity_confinement.data, kernel_vector(context.reconstructed_velocity_tangent_), kernel_vector(context.vorticity_tangent_), kernel_vector(context.vorticity_normal_tangent_), parameter_tangent.vorticity_confinement.data, kernel_vector(context.confined_velocity_tangent_));
        cuda_kernel::launch_xsph_jvp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.confined_velocities), kernel_vector(context.current_position_tangent_), kernel_vector(context.confined_velocity_tangent_), kernel_neighborhood(step_cache.neighborhood), parameters.xsph_viscosity.data, parameter_tangent.xsph_viscosity.data, kernel_vector(next_state_tangent.velocities));
        sph::copy_vector(*context.resource, context.current_position_tangent_, next_state_tangent.positions);
    }

    void Model::vjp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const {
        const std::uint32_t count = configuration.fluid.particle_count;
        sph::copy_vector_adjoint(*context.resource, next_state_adjoint.positions, context.current_position_adjoint_);
        sph::zero_vector_adjoint(*context.resource, context.confined_velocity_adjoint_);
        cuda_kernel::launch_xsph_vjp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.confined_velocities), kernel_neighborhood(step_cache.neighborhood), parameters.xsph_viscosity.data, kernel_vector(next_state_adjoint.velocities), kernel_vector(context.current_position_adjoint_), kernel_vector(context.confined_velocity_adjoint_), parameter_adjoint.xsph_viscosity.data);

        sph::zero_vector_adjoint(*context.resource, context.reconstructed_velocity_adjoint_);
        sph::zero_vector_adjoint(*context.resource, context.vorticity_adjoint_);
        sph::zero_vector_adjoint(*context.resource, context.vorticity_normal_adjoint_);
        cuda_kernel::launch_confinement_vjp(context.resource->native_stream, count, configuration.fluid.time_step, kernel_vector(step_cache.vorticities), kernel_vector(step_cache.vorticity_normals), parameters.vorticity_confinement.data, kernel_vector(context.confined_velocity_adjoint_), kernel_vector(context.reconstructed_velocity_adjoint_), kernel_vector(context.vorticity_adjoint_), kernel_vector(context.vorticity_normal_adjoint_), parameter_adjoint.vorticity_confinement.data);
        cuda_kernel::launch_normal_vjp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.vorticities), kernel_neighborhood(step_cache.neighborhood), step_cache.vorticity_magnitudes.values.data, kernel_vector(step_cache.vorticity_normals), step_cache.vorticity_normalizers.values.data, kernel_vector(context.vorticity_normal_adjoint_), kernel_vector(context.current_position_adjoint_), kernel_vector(context.vorticity_adjoint_));
        cuda_kernel::launch_vorticity_vjp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(step_cache.corrected_positions), kernel_vector(step_cache.reconstructed_velocities), kernel_neighborhood(step_cache.neighborhood), kernel_vector(context.vorticity_adjoint_), kernel_vector(context.current_position_adjoint_), kernel_vector(context.reconstructed_velocity_adjoint_));
        cuda_kernel::launch_reconstruct_vjp(context.resource->native_stream, count, 1.0F / configuration.fluid.time_step, kernel_vector(context.reconstructed_velocity_adjoint_), kernel_vector(previous_state_adjoint.positions), kernel_vector(context.current_position_adjoint_));

        for (std::size_t segment = step_cache.checkpoints.size(); segment-- > 1u;) {
            const StepCache::IterationCheckpoint& start = step_cache.checkpoints[segment - 1u];
            const StepCache::IterationCheckpoint& end = step_cache.checkpoints[segment];
            const std::uint32_t segment_length = end.iteration - start.iteration;
            sph::copy_vector(*context.resource, start.positions, context.segment_position_history_[0]);
            for (std::uint32_t local_iteration = 0u; local_iteration < segment_length; ++local_iteration) {
                ExecutionContext::IterationPrimal& iteration = context.segment_iteration_history_[local_iteration];
                sph::pbf_density_forward_frozen(*context.resource, configuration.fluid, context.boundary, step_cache.predicted_positions, context.segment_position_history_[local_iteration], parameters.particles, step_cache.neighborhood, iteration.densities);
                cuda_kernel::launch_lambda_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(context.segment_position_history_[local_iteration]), kernel_parameters(parameters.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), iteration.densities.values.data, parameters.relaxation.data, kernel_vector(iteration.gradient_sums), iteration.denominators.values.data, iteration.lambdas.values.data);
                cuda_kernel::launch_correction_forward(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(context.segment_position_history_[local_iteration]), kernel_parameters(parameters.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), iteration.lambdas.values.data, parameters.artificial_pressure_strength.data, parameters.artificial_pressure_exponent.data, parameters.artificial_pressure_radius.data, kernel_vector(iteration.corrections));
                cuda_kernel::launch_project_forward(context.resource->native_stream, count, kernel_domain(configuration, state.step_index + 1u), kernel_vector(context.segment_position_history_[local_iteration]), kernel_vector(iteration.corrections), iteration.collision_masks.data, kernel_vector(context.segment_position_history_[local_iteration + 1u]));
            }
            for (std::uint32_t local_iteration = segment_length; local_iteration-- > 0u;) {
                ExecutionContext::IterationPrimal& iteration = context.segment_iteration_history_[local_iteration];
                sph::zero_vector_adjoint(*context.resource, context.next_position_adjoint_);
                sph::zero_vector_adjoint(*context.resource, context.correction_adjoint_);
                sph::zero_scalar_adjoint(*context.resource, context.lambda_adjoint_);
                sph::zero_scalar_adjoint(*context.resource, context.density_adjoint_);
                cuda_kernel::launch_project_vjp(context.resource->native_stream, count, iteration.collision_masks.data, kernel_vector(context.current_position_adjoint_), kernel_vector(context.next_position_adjoint_), kernel_vector(context.correction_adjoint_));
                cuda_kernel::launch_correction_vjp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(context.segment_position_history_[local_iteration]), kernel_parameters(parameters.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), iteration.lambdas.values.data, parameters.artificial_pressure_strength.data, parameters.artificial_pressure_exponent.data, parameters.artificial_pressure_radius.data, kernel_vector(context.correction_adjoint_), kernel_vector(context.next_position_adjoint_), context.lambda_adjoint_.values.data, kernel_parameters(parameter_adjoint.particles), parameter_adjoint.artificial_pressure_strength.data, parameter_adjoint.artificial_pressure_exponent.data, parameter_adjoint.artificial_pressure_radius.data);
                cuda_kernel::launch_lambda_vjp(context.resource->native_stream, count, configuration.fluid.support_radius, kernel_vector(step_cache.predicted_positions), kernel_vector(context.segment_position_history_[local_iteration]), kernel_parameters(parameters.particles), kernel_neighborhood(step_cache.neighborhood), kernel_boundary(context.boundary, step_cache.neighborhood), iteration.densities.values.data, kernel_vector(iteration.gradient_sums), iteration.denominators.values.data, context.lambda_adjoint_.values.data, kernel_vector(context.next_position_adjoint_), context.density_adjoint_.values.data, kernel_parameters(parameter_adjoint.particles), parameter_adjoint.relaxation.data);
                sph::pbf_density_vjp_frozen(*context.resource, configuration.fluid, context.boundary, step_cache.predicted_positions, context.segment_position_history_[local_iteration], parameters.particles, step_cache.neighborhood, context.density_adjoint_, context.next_position_adjoint_, parameter_adjoint.particles);
                std::swap(context.current_position_adjoint_, context.next_position_adjoint_);
            }
        }
        cuda_kernel::launch_predict_vjp(context.resource->native_stream, count, configuration.fluid.time_step, kernel_vector(context.current_position_adjoint_), kernel_vector(previous_state_adjoint.positions), kernel_vector(previous_state_adjoint.velocities), kernel_vector(control_adjoint.external_accelerations));
    }

} // namespace xayah::fluid::pbf
