module;

#include "sph.h"

module xayah.fluid.sph;

import std;
import xayah.cuda;
import xayah.fluid.data;

namespace xayah::fluid::sph {

    namespace {

        cuda_kernel::ConstVector kernel_vector(const VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernel::Vector kernel_vector(VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernel::ConstVectorAdjoint kernel_vector(const VectorAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernel::VectorAdjoint kernel_vector(VectorAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernel::Boundary kernel_boundary(const DeviceBoundary& boundary, const Neighborhood& neighborhood) {
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

        cuda_kernel::Neighborhood kernel_neighborhood(const Neighborhood& neighborhood) {
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

        cuda_kernel::Domain kernel_domain(const Configuration& configuration) {
            return {
                .minimum_x = configuration.domain.minimum.x,
                .minimum_y = configuration.domain.minimum.y,
                .minimum_z = configuration.domain.minimum.z,
                .maximum_x = configuration.domain.maximum.x,
                .maximum_y = configuration.domain.maximum.y,
                .maximum_z = configuration.domain.maximum.z,
                .velocity_x = configuration.domain.velocity.x,
                .velocity_y = configuration.domain.velocity.y,
                .velocity_z = configuration.domain.velocity.z,
                .no_slip = configuration.domain.no_slip ? 1u : 0u,
            };
        }

        cuda_kernel::Domain kernel_domain(const Configuration& configuration, const std::uint64_t step_index) {
            cuda_kernel::Domain domain = kernel_domain(configuration);
            const float time = static_cast<float>(step_index) * configuration.time_step;
            domain.minimum_x += time * domain.velocity_x;
            domain.minimum_y += time * domain.velocity_y;
            domain.minimum_z += time * domain.velocity_z;
            domain.maximum_x += time * domain.velocity_x;
            domain.maximum_y += time * domain.velocity_y;
            domain.maximum_z += time * domain.velocity_z;
            domain.minimum_x += configuration.particle_radius;
            domain.minimum_y += configuration.particle_radius;
            domain.minimum_z += configuration.particle_radius;
            domain.maximum_x -= configuration.particle_radius;
            domain.maximum_y -= configuration.particle_radius;
            domain.maximum_z -= configuration.particle_radius;
            return domain;
        }

        cuda_kernel::ParticleParameters kernel_parameters(const ParticleParameters& parameters) {
            return {.masses = parameters.masses.data, .rest_densities = parameters.rest_densities.data, .viscosities = parameters.viscosities.data, .surface_tensions = parameters.surface_tensions.data};
        }

        cuda_kernel::ParticleParameterTangent kernel_parameters(const ParticleParameterTangent& parameters) {
            return {.masses = parameters.masses.data, .rest_densities = parameters.rest_densities.data, .viscosities = parameters.viscosities.data, .surface_tensions = parameters.surface_tensions.data};
        }

        cuda_kernel::ParticleParameterAdjoint kernel_parameters(ParticleParameterAdjoint& parameters) {
            return {.masses = parameters.masses.data, .rest_densities = parameters.rest_densities.data, .viscosities = parameters.viscosities.data, .surface_tensions = parameters.surface_tensions.data};
        }

    } // namespace

    VectorField allocate_vector_field(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {.x = cuda::Buffer<float>(resource, count), .y = cuda::Buffer<float>(resource, count), .z = cuda::Buffer<float>(resource, count)};
    }

    VectorAdjointField allocate_vector_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {.x = cuda::Buffer<double>(resource, count), .y = cuda::Buffer<double>(resource, count), .z = cuda::Buffer<double>(resource, count)};
    }

    ScalarField allocate_scalar_field(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {.values = cuda::Buffer<float>(resource, count)};
    }

    ScalarAdjointField allocate_scalar_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {.values = cuda::Buffer<double>(resource, count)};
    }

    ParticleParameters allocate_particle_parameters(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {
            .masses           = cuda::Buffer<float>(resource, count),
            .rest_densities   = cuda::Buffer<float>(resource, count),
            .viscosities      = cuda::Buffer<float>(resource, count),
            .surface_tensions = cuda::Buffer<float>(resource, count),
        };
    }

    ParticleParameterTangent allocate_particle_parameter_tangent(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {
            .masses           = cuda::Buffer<float>(resource, count),
            .rest_densities   = cuda::Buffer<float>(resource, count),
            .viscosities      = cuda::Buffer<float>(resource, count),
            .surface_tensions = cuda::Buffer<float>(resource, count),
        };
    }

    ParticleParameterAdjoint allocate_particle_parameter_adjoint(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {
            .masses           = cuda::Buffer<double>(resource, count),
            .rest_densities   = cuda::Buffer<double>(resource, count),
            .viscosities      = cuda::Buffer<double>(resource, count),
            .surface_tensions = cuda::Buffer<double>(resource, count),
        };
    }

    State allocate_state(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {.positions = allocate_vector_field(resource, count), .velocities = allocate_vector_field(resource, count)};
    }

    Control allocate_control(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        Control control{.external_accelerations = allocate_vector_field(resource, count)};
        zero_vector(*resource, control.external_accelerations);
        return control;
    }

    StateTangent allocate_state_tangent(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        StateTangent tangent{.positions = allocate_vector_field(resource, count), .velocities = allocate_vector_field(resource, count)};
        zero_vector(*resource, tangent.positions);
        zero_vector(*resource, tangent.velocities);
        return tangent;
    }

    ControlTangent allocate_control_tangent(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        ControlTangent tangent{.external_accelerations = allocate_vector_field(resource, count)};
        zero_vector(*resource, tangent.external_accelerations);
        return tangent;
    }

    StateAdjoint allocate_state_adjoint(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        StateAdjoint adjoint{.positions = allocate_vector_adjoint_field(resource, count), .velocities = allocate_vector_adjoint_field(resource, count)};
        zero_vector_adjoint(*resource, adjoint.positions);
        zero_vector_adjoint(*resource, adjoint.velocities);
        return adjoint;
    }

    ControlAdjoint allocate_control_adjoint(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        ControlAdjoint adjoint{.external_accelerations = allocate_vector_adjoint_field(resource, count)};
        zero_vector_adjoint(*resource, adjoint.external_accelerations);
        return adjoint;
    }

    DeviceBoundary allocate_boundary(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration) {
        DeviceBoundary boundary{
            .positions  = allocate_vector_field(resource, configuration.boundary_particles.size()),
            .velocities = allocate_vector_field(resource, configuration.boundary_particles.size()),
            .volumes    = allocate_scalar_field(resource, configuration.boundary_particles.size()),
        };
        std::vector<float> position_x(configuration.boundary_particles.size());
        std::vector<float> position_y(configuration.boundary_particles.size());
        std::vector<float> position_z(configuration.boundary_particles.size());
        std::vector<float> velocity_x(configuration.boundary_particles.size());
        std::vector<float> velocity_y(configuration.boundary_particles.size());
        std::vector<float> velocity_z(configuration.boundary_particles.size());
        std::vector<float> volumes(configuration.boundary_particles.size());
        for (std::size_t index = 0; index < configuration.boundary_particles.size(); ++index) {
            position_x[index] = configuration.boundary_particles[index].position.x;
            position_y[index] = configuration.boundary_particles[index].position.y;
            position_z[index] = configuration.boundary_particles[index].position.z;
            velocity_x[index] = configuration.boundary_particles[index].velocity.x;
            velocity_y[index] = configuration.boundary_particles[index].velocity.y;
            velocity_z[index] = configuration.boundary_particles[index].velocity.z;
            volumes[index]    = configuration.boundary_particles[index].volume;
        }
        if (!volumes.empty()) {
            resource->copy_from_host(boundary.positions.x.data, position_x.data(), position_x.size() * sizeof(float));
            resource->copy_from_host(boundary.positions.y.data, position_y.data(), position_y.size() * sizeof(float));
            resource->copy_from_host(boundary.positions.z.data, position_z.data(), position_z.size() * sizeof(float));
            resource->copy_from_host(boundary.velocities.x.data, velocity_x.data(), velocity_x.size() * sizeof(float));
            resource->copy_from_host(boundary.velocities.y.data, velocity_y.data(), velocity_y.size() * sizeof(float));
            resource->copy_from_host(boundary.velocities.z.data, velocity_z.data(), velocity_z.size() * sizeof(float));
            resource->copy_from_host(boundary.volumes.values.data, volumes.data(), volumes.size() * sizeof(float));
        }
        return boundary;
    }

    NeighborSearch allocate_neighbor_search(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration) {
        std::size_t sort_bytes;
        std::size_t boundary_sort_bytes;
        cuda_kernel::query_neighbor_storage(configuration.particle_count, sort_bytes);
        cuda_kernel::query_neighbor_storage(static_cast<std::uint32_t>(configuration.boundary_particles.size()), boundary_sort_bytes);
        return {
            .unsorted_keys             = cuda::Buffer<std::uint64_t>(resource, configuration.particle_count),
            .unsorted_particle_indices = cuda::Buffer<std::uint32_t>(resource, configuration.particle_count),
            .unsorted_boundary_keys    = cuda::Buffer<std::uint64_t>(resource, configuration.boundary_particles.size()),
            .unsorted_boundary_indices = cuda::Buffer<std::uint32_t>(resource, configuration.boundary_particles.size()),
            .sort_scratch               = cuda::Buffer<std::byte>(resource, sort_bytes),
            .boundary_sort_scratch      = cuda::Buffer<std::byte>(resource, boundary_sort_bytes),
        };
    }

    Neighborhood allocate_neighborhood(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration) {
        const std::uint32_t cells_x = static_cast<std::uint32_t>(std::ceil((configuration.domain.maximum.x - configuration.domain.minimum.x) / configuration.support_radius));
        const std::uint32_t cells_y = static_cast<std::uint32_t>(std::ceil((configuration.domain.maximum.y - configuration.domain.minimum.y) / configuration.support_radius));
        const std::uint32_t cells_z = static_cast<std::uint32_t>(std::ceil((configuration.domain.maximum.z - configuration.domain.minimum.z) / configuration.support_radius));
        const std::size_t cell_count = static_cast<std::size_t>(cells_x) * cells_y * cells_z;
        return {
            .sorted_keys             = cuda::Buffer<std::uint64_t>(resource, configuration.particle_count),
            .sorted_particle_indices = cuda::Buffer<std::uint32_t>(resource, configuration.particle_count),
            .cell_offsets            = cuda::Buffer<std::uint32_t>(resource, cell_count + 1u),
            .sorted_boundary_keys    = cuda::Buffer<std::uint64_t>(resource, configuration.boundary_particles.size()),
            .sorted_boundary_indices = cuda::Buffer<std::uint32_t>(resource, configuration.boundary_particles.size()),
            .boundary_cell_offsets   = cuda::Buffer<std::uint32_t>(resource, cell_count + 1u),
            .cell_resolution         = {cells_x, cells_y, cells_z},
            .cell_origin             = configuration.domain.minimum,
            .cell_size               = configuration.support_radius,
            .boundary_time           = 0.0F,
        };
    }

    void copy_state(cuda::Resource& resource, const State& source, State& destination) {
        cuda_kernel::launch_copy_vector(resource.native_stream, static_cast<std::uint32_t>(source.positions.x.size), kernel_vector(source.positions), kernel_vector(destination.positions));
        cuda_kernel::launch_copy_vector(resource.native_stream, static_cast<std::uint32_t>(source.velocities.x.size), kernel_vector(source.velocities), kernel_vector(destination.velocities));
        destination.step_index = source.step_index;
    }

    void copy_state_tangent(cuda::Resource& resource, const StateTangent& source, StateTangent& destination) {
        cuda_kernel::launch_copy_vector(resource.native_stream, static_cast<std::uint32_t>(source.positions.x.size), kernel_vector(source.positions), kernel_vector(destination.positions));
        cuda_kernel::launch_copy_vector(resource.native_stream, static_cast<std::uint32_t>(source.velocities.x.size), kernel_vector(source.velocities), kernel_vector(destination.velocities));
    }

    void copy_state_adjoint(cuda::Resource& resource, const StateAdjoint& source, StateAdjoint& destination) {
        cuda_kernel::launch_copy_vector_adjoint(resource.native_stream, static_cast<std::uint32_t>(source.positions.x.size), kernel_vector(source.positions), kernel_vector(destination.positions));
        cuda_kernel::launch_copy_vector_adjoint(resource.native_stream, static_cast<std::uint32_t>(source.velocities.x.size), kernel_vector(source.velocities), kernel_vector(destination.velocities));
    }

    void accumulate_state_adjoint(cuda::Resource& resource, const StateAdjoint& source, StateAdjoint& destination) {
        cuda_kernel::launch_accumulate_vector_adjoint(resource.native_stream, static_cast<std::uint32_t>(source.positions.x.size), kernel_vector(source.positions), kernel_vector(destination.positions));
        cuda_kernel::launch_accumulate_vector_adjoint(resource.native_stream, static_cast<std::uint32_t>(source.velocities.x.size), kernel_vector(source.velocities), kernel_vector(destination.velocities));
    }

    void copy_vector(cuda::Resource& resource, const VectorField& source, VectorField& destination) {
        cuda_kernel::launch_copy_vector(resource.native_stream, static_cast<std::uint32_t>(source.x.size), kernel_vector(source), kernel_vector(destination));
    }

    void copy_scalar(cuda::Resource& resource, const ScalarField& source, ScalarField& destination) {
        cuda_kernel::launch_copy_scalar(resource.native_stream, static_cast<std::uint32_t>(source.values.size), source.values.data, destination.values.data);
    }

    void copy_vector_adjoint(cuda::Resource& resource, const VectorAdjointField& source, VectorAdjointField& destination) {
        cuda_kernel::launch_copy_vector_adjoint(resource.native_stream, static_cast<std::uint32_t>(source.x.size), kernel_vector(source), kernel_vector(destination));
    }

    void copy_scalar_adjoint(cuda::Resource& resource, const ScalarAdjointField& source, ScalarAdjointField& destination) {
        cuda_kernel::launch_copy_scalar_adjoint(resource.native_stream, static_cast<std::uint32_t>(source.values.size), source.values.data, destination.values.data);
    }

    void accumulate_vector_adjoint(cuda::Resource& resource, const VectorAdjointField& source, VectorAdjointField& destination) {
        cuda_kernel::launch_accumulate_vector_adjoint(resource.native_stream, static_cast<std::uint32_t>(source.x.size), kernel_vector(source), kernel_vector(destination));
    }

    void accumulate_scalar_adjoint(cuda::Resource& resource, const ScalarAdjointField& source, ScalarAdjointField& destination) {
        cuda_kernel::launch_accumulate_scalar_adjoint(resource.native_stream, static_cast<std::uint32_t>(source.values.size), source.values.data, destination.values.data);
    }

    void zero_vector(cuda::Resource& resource, VectorField& field) {
        resource.zero(field.x.data, field.x.size * sizeof(float));
        resource.zero(field.y.data, field.y.size * sizeof(float));
        resource.zero(field.z.data, field.z.size * sizeof(float));
    }

    void zero_vector_adjoint(cuda::Resource& resource, VectorAdjointField& field) {
        resource.zero(field.x.data, field.x.size * sizeof(double));
        resource.zero(field.y.data, field.y.size * sizeof(double));
        resource.zero(field.z.data, field.z.size * sizeof(double));
    }

    void zero_scalar(cuda::Resource& resource, ScalarField& field) {
        resource.zero(field.values.data, field.values.size * sizeof(float));
    }

    void zero_scalar_adjoint(cuda::Resource& resource, ScalarAdjointField& field) {
        resource.zero(field.values.data, field.values.size * sizeof(double));
    }

    void build_neighborhood(cuda::Resource& resource, const Configuration& configuration, const std::uint64_t step_index, const DeviceBoundary& boundary, const VectorField& positions, NeighborSearch& search, Neighborhood& neighborhood) {
        neighborhood.boundary_time = static_cast<float>(step_index) * configuration.time_step;
        neighborhood.cell_origin = {
            .x = configuration.domain.minimum.x + neighborhood.boundary_time * configuration.domain.velocity.x,
            .y = configuration.domain.minimum.y + neighborhood.boundary_time * configuration.domain.velocity.y,
            .z = configuration.domain.minimum.z + neighborhood.boundary_time * configuration.domain.velocity.z,
        };
        cuda_kernel::launch_build_neighborhood(resource.native_stream, configuration.particle_count, static_cast<std::uint32_t>(configuration.boundary_particles.size()), configuration.support_radius, configuration.time_step, step_index, kernel_domain(configuration), kernel_vector(positions), kernel_vector(boundary.positions), kernel_vector(boundary.velocities), search.unsorted_keys.data, search.unsorted_particle_indices.data, search.unsorted_boundary_keys.data, search.unsorted_boundary_indices.data, search.sort_scratch.data, search.sort_scratch.size, search.boundary_sort_scratch.data, search.boundary_sort_scratch.size, neighborhood.sorted_keys.data, neighborhood.sorted_particle_indices.data, neighborhood.cell_offsets.data, neighborhood.sorted_boundary_keys.data, neighborhood.sorted_boundary_indices.data, neighborhood.boundary_cell_offsets.data);
    }

    void density_forward(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities) {
        cuda_kernel::launch_density_forward(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data);
    }

    void density_jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent) {
        cuda_kernel::launch_density_jvp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_vector(positions), kernel_vector(position_tangent), kernel_parameters(parameters), kernel_parameters(parameter_tangent), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), density_tangent.values.data);
    }

    void density_vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
        cuda_kernel::launch_density_vjp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), density_adjoint.values.data, kernel_vector(position_adjoint), kernel_parameters(parameter_adjoint));
    }

    void density_forward_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities) {
        cuda_kernel::launch_density_forward(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(topology_positions), kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data);
    }

    void density_jvp_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent) {
        cuda_kernel::launch_density_jvp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(topology_positions), kernel_vector(positions), kernel_vector(position_tangent), kernel_parameters(parameters), kernel_parameters(parameter_tangent), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), density_tangent.values.data);
    }

    void density_vjp_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
        cuda_kernel::launch_density_vjp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(topology_positions), kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), density_adjoint.values.data, kernel_vector(position_adjoint), kernel_parameters(parameter_adjoint));
    }

    void pbf_density_forward(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities) {
        cuda_kernel::launch_pbf_density_forward(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data);
    }

    void pbf_density_jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent) {
        cuda_kernel::launch_pbf_density_jvp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_vector(positions), kernel_vector(position_tangent), kernel_parameters(parameters), kernel_parameters(parameter_tangent), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), density_tangent.values.data);
    }

    void pbf_density_vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
        cuda_kernel::launch_pbf_density_vjp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), density_adjoint.values.data, kernel_vector(position_adjoint), kernel_parameters(parameter_adjoint));
    }

    void pbf_density_forward_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities) {
        cuda_kernel::launch_pbf_density_forward(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(topology_positions), kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data);
    }

    void pbf_density_jvp_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent) {
        cuda_kernel::launch_pbf_density_jvp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(topology_positions), kernel_vector(positions), kernel_vector(position_tangent), kernel_parameters(parameters), kernel_parameters(parameter_tangent), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), density_tangent.values.data);
    }

    void pbf_density_vjp_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
        cuda_kernel::launch_pbf_density_vjp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(topology_positions), kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), density_adjoint.values.data, kernel_vector(position_adjoint), kernel_parameters(parameter_adjoint));
    }

    void non_pressure_forward(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const State& state, const Control& control, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, VectorField& accelerations) {
        cuda_kernel::launch_non_pressure_forward(resource.native_stream, configuration.particle_count, configuration.support_radius, configuration.gravity.x, configuration.gravity.y, configuration.gravity.z, kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(control.external_accelerations), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data, kernel_vector(accelerations));
    }

    void non_pressure_jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const State& state, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& parameter_tangent, const ScalarField& density_tangent, VectorField& acceleration_tangent) {
        cuda_kernel::launch_non_pressure_jvp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(control_tangent.external_accelerations), kernel_vector(state_tangent.positions), kernel_vector(state_tangent.velocities), kernel_parameters(parameters), kernel_parameters(parameter_tangent), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data, density_tangent.values.data, kernel_vector(acceleration_tangent));
    }

    void non_pressure_vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const State& state, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const VectorAdjointField& acceleration_adjoint, StateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, ScalarAdjointField& density_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
        cuda_kernel::launch_non_pressure_vjp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(state.positions), kernel_vector(state.velocities), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data, kernel_vector(acceleration_adjoint), kernel_vector(state_adjoint.positions), kernel_vector(state_adjoint.velocities), kernel_vector(control_adjoint.external_accelerations), density_adjoint.values.data, kernel_parameters(parameter_adjoint));
    }

    void pressure_acceleration_forward(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, VectorField& accelerations) {
        cuda_kernel::launch_pressure_acceleration_forward(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data, pressures.values.data, kernel_vector(accelerations));
    }

    void pressure_acceleration_jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, const VectorField& position_tangent, const ParticleParameterTangent& parameter_tangent, const ScalarField& density_tangent, const ScalarField& pressure_tangent, VectorField& acceleration_tangent) {
        cuda_kernel::launch_pressure_acceleration_jvp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_vector(position_tangent), kernel_parameters(parameters), kernel_parameters(parameter_tangent), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data, density_tangent.values.data, pressures.values.data, pressure_tangent.values.data, kernel_vector(acceleration_tangent));
    }

    void pressure_acceleration_vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, const VectorAdjointField& acceleration_adjoint, VectorAdjointField& position_adjoint, ScalarAdjointField& density_adjoint, ScalarAdjointField& pressure_adjoint, ParticleParameterAdjoint& parameter_adjoint) {
        cuda_kernel::launch_pressure_acceleration_vjp(resource.native_stream, configuration.particle_count, configuration.support_radius, kernel_vector(positions), kernel_parameters(parameters), kernel_neighborhood(neighborhood), kernel_boundary(boundary, neighborhood), densities.values.data, pressures.values.data, kernel_vector(acceleration_adjoint), kernel_vector(position_adjoint), density_adjoint.values.data, pressure_adjoint.values.data, kernel_parameters(parameter_adjoint));
    }

    void add_accelerations(cuda::Resource& resource, const VectorField& first, const VectorField& second, VectorField& output) {
        cuda_kernel::launch_add_acceleration(resource.native_stream, static_cast<std::uint32_t>(first.x.size), kernel_vector(first), kernel_vector(second), kernel_vector(output));
    }

    void integrate_forward(cuda::Resource& resource, const Configuration& configuration, const State& state, const VectorField& accelerations, State& next_state) {
        cuda_kernel::launch_integrate_forward(resource.native_stream, configuration.particle_count, configuration.time_step, kernel_domain(configuration, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(accelerations), kernel_vector(next_state.positions), kernel_vector(next_state.velocities));
        next_state.step_index = state.step_index + 1u;
    }

    void integrate_jvp(cuda::Resource& resource, const Configuration& configuration, const State& state, const VectorField& accelerations, const StateTangent& state_tangent, const VectorField& acceleration_tangent, StateTangent& next_state_tangent) {
        cuda_kernel::launch_integrate_jvp(resource.native_stream, configuration.particle_count, configuration.time_step, kernel_domain(configuration, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(accelerations), kernel_vector(state_tangent.positions), kernel_vector(state_tangent.velocities), kernel_vector(acceleration_tangent), kernel_vector(next_state_tangent.positions), kernel_vector(next_state_tangent.velocities));
    }

    void integrate_vjp(cuda::Resource& resource, const Configuration& configuration, const State& state, const VectorField& accelerations, const StateAdjoint& next_state_adjoint, StateAdjoint& state_adjoint, VectorAdjointField& acceleration_adjoint) {
        cuda_kernel::launch_integrate_vjp(resource.native_stream, configuration.particle_count, configuration.time_step, kernel_domain(configuration, state.step_index), kernel_vector(state.positions), kernel_vector(state.velocities), kernel_vector(accelerations), kernel_vector(next_state_adjoint.positions), kernel_vector(next_state_adjoint.velocities), kernel_vector(state_adjoint.positions), kernel_vector(state_adjoint.velocities), kernel_vector(acceleration_adjoint));
    }

} // namespace xayah::fluid::sph
