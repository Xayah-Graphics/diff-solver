module;

#include "grid.h"
#include <cuda_runtime_api.h>

module xayah.fluid.grid;

import std;
import xayah.cuda;
import xayah.fluid.data;

namespace xayah::fluid::grid {

    namespace {

        cuda_kernels::Configuration kernel_configuration(const Configuration& configuration) {
            return {
                .nx                       = configuration.resolution[0],
                .ny                       = configuration.resolution[1],
                .nz                       = configuration.resolution[2],
                .origin_x                 = configuration.origin.x,
                .origin_y                 = configuration.origin.y,
                .origin_z                 = configuration.origin.z,
                .cell_size                = configuration.cell_size,
                .time_step                = configuration.time_step,
                .pressure_iterations      = configuration.pressure_iterations,
                .extrapolation_iterations = configuration.extrapolation_iterations,
                .pressure_anchor          = configuration.pressure_anchor.value_or(0u),
                .pressure_anchor_enabled  = configuration.pressure_anchor.has_value(),
            };
        }

        cuda_kernels::ConstVectorField view(const VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::VectorField view(VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::ConstVectorAdjointField view(const VectorAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::VectorAdjointField view(VectorAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::ConstMatrixField view(const MatrixField* field) {
            if (field == nullptr) return {};
            return {.values = {field->xx.data, field->xy.data, field->xz.data, field->yx.data, field->yy.data, field->yz.data, field->zx.data, field->zy.data, field->zz.data}};
        }

        cuda_kernels::MatrixAdjointField view(MatrixAdjointField* field) {
            if (field == nullptr) return {};
            return {.values = {field->xx.data, field->xy.data, field->xz.data, field->yx.data, field->yy.data, field->yz.data, field->zx.data, field->zy.data, field->zz.data}};
        }

        cuda_kernels::ConstStaggeredField view(const StaggeredField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::StaggeredField view(StaggeredField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::ConstStaggeredAdjointField view(const StaggeredAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::StaggeredAdjointField view(StaggeredAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        void zero_staggered(cuda::Resource& resource, StaggeredField& field) {
            resource.zero(field.x.data, field.x.size * sizeof(float));
            resource.zero(field.y.data, field.y.size * sizeof(float));
            resource.zero(field.z.data, field.z.size * sizeof(float));
        }

        void zero_staggered(cuda::Resource& resource, StaggeredAdjointField& field) {
            resource.zero(field.x.data, field.x.size * sizeof(double));
            resource.zero(field.y.data, field.y.size * sizeof(double));
            resource.zero(field.z.data, field.z.size * sizeof(double));
        }

    } // namespace

    std::array<std::size_t, 3u> face_counts(const Configuration& configuration) {
        const std::size_t nx = configuration.resolution[0];
        const std::size_t ny = configuration.resolution[1];
        const std::size_t nz = configuration.resolution[2];
        return {(nx + 1u) * ny * nz, nx * (ny + 1u) * nz, nx * ny * (nz + 1u)};
    }

    std::size_t cell_count(const Configuration& configuration) {
        return static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * configuration.resolution[2];
    }

    std::size_t total_face_count(const Configuration& configuration) {
        const std::array counts = face_counts(configuration);
        return counts[0] + counts[1] + counts[2];
    }

    DeviceDomain allocate_domain(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration) {
        DeviceDomain domain{.cell_types = cuda::Buffer<std::uint32_t>(resource, cell_count(configuration)), .solid_velocity = allocate_staggered_field(resource, configuration)};
        std::vector<std::uint32_t> cell_types(configuration.solid_cell_mask.size());
        for (std::size_t cell = 0; cell < cell_types.size(); ++cell) cell_types[cell] = configuration.solid_cell_mask[cell] == 0u ? static_cast<std::uint32_t>(CellType::air) : static_cast<std::uint32_t>(CellType::solid);
        resource->copy_from_host(domain.cell_types.data, cell_types.data(), cell_types.size() * sizeof(std::uint32_t));
        resource->copy_from_host(domain.solid_velocity.x.data, configuration.solid_face_velocities[0].data(), configuration.solid_face_velocities[0].size() * sizeof(float));
        resource->copy_from_host(domain.solid_velocity.y.data, configuration.solid_face_velocities[1].data(), configuration.solid_face_velocities[1].size() * sizeof(float));
        resource->copy_from_host(domain.solid_velocity.z.data, configuration.solid_face_velocities[2].data(), configuration.solid_face_velocities[2].size() * sizeof(float));
        return domain;
    }

    VectorField allocate_vector_field(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {.x = cuda::Buffer<float>(resource, count), .y = cuda::Buffer<float>(resource, count), .z = cuda::Buffer<float>(resource, count)};
    }

    VectorAdjointField allocate_vector_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {.x = cuda::Buffer<double>(resource, count), .y = cuda::Buffer<double>(resource, count), .z = cuda::Buffer<double>(resource, count)};
    }

    MatrixField allocate_matrix_field(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {
            .xx = cuda::Buffer<float>(resource, count), .xy = cuda::Buffer<float>(resource, count), .xz = cuda::Buffer<float>(resource, count),
            .yx = cuda::Buffer<float>(resource, count), .yy = cuda::Buffer<float>(resource, count), .yz = cuda::Buffer<float>(resource, count),
            .zx = cuda::Buffer<float>(resource, count), .zy = cuda::Buffer<float>(resource, count), .zz = cuda::Buffer<float>(resource, count),
        };
    }

    MatrixAdjointField allocate_matrix_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, const std::size_t count) {
        return {
            .xx = cuda::Buffer<double>(resource, count), .xy = cuda::Buffer<double>(resource, count), .xz = cuda::Buffer<double>(resource, count),
            .yx = cuda::Buffer<double>(resource, count), .yy = cuda::Buffer<double>(resource, count), .yz = cuda::Buffer<double>(resource, count),
            .zx = cuda::Buffer<double>(resource, count), .zy = cuda::Buffer<double>(resource, count), .zz = cuda::Buffer<double>(resource, count),
        };
    }

    StaggeredField allocate_staggered_field(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration) {
        const std::array counts = face_counts(configuration);
        return {.x = cuda::Buffer<float>(resource, counts[0]), .y = cuda::Buffer<float>(resource, counts[1]), .z = cuda::Buffer<float>(resource, counts[2])};
    }

    StaggeredAdjointField allocate_staggered_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration) {
        const std::array counts = face_counts(configuration);
        return {.x = cuda::Buffer<double>(resource, counts[0]), .y = cuda::Buffer<double>(resource, counts[1]), .z = cuda::Buffer<double>(resource, counts[2])};
    }

    StepCache allocate_step_cache(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration, const std::uint32_t particle_count) {
        const std::size_t cells         = cell_count(configuration);
        const std::size_t faces         = total_face_count(configuration);
        const std::size_t contributions = static_cast<std::size_t>(particle_count) * 81u;
        StepCache cache{
            .transfer = {
                .unsorted_keys  = cuda::Buffer<std::uint64_t>(resource, contributions),
                .sorted_keys    = cuda::Buffer<std::uint64_t>(resource, contributions),
                .unsorted_ids   = cuda::Buffer<std::uint32_t>(resource, contributions),
                .sorted_ids     = cuda::Buffer<std::uint32_t>(resource, contributions),
                .unsorted_cell_keys = cuda::Buffer<std::uint64_t>(resource, particle_count),
                .sorted_cell_keys   = cuda::Buffer<std::uint64_t>(resource, particle_count),
                .unsorted_cell_ids  = cuda::Buffer<std::uint32_t>(resource, particle_count),
                .sorted_cell_ids    = cuda::Buffer<std::uint32_t>(resource, particle_count),
                .face_mass      = cuda::Buffer<float>(resource, faces),
                .old_velocity   = allocate_staggered_field(resource, configuration),
                .forced_velocity = allocate_staggered_field(resource, configuration),
                .sort_scratch   = cuda::Buffer<std::byte>(resource, std::max(cuda_kernels::sort_scratch_bytes(contributions), cuda_kernels::reduction_scratch_bytes(particle_count))),
                .reduction_values = cuda::Buffer<double>(resource, particle_count),
                .reduction_result = cuda::Buffer<double>(resource, 1u),
            },
            .projection = {
                .cell_types        = cuda::Buffer<std::uint32_t>(resource, cells),
                .divergence        = cuda::Buffer<float>(resource, cells),
                .pressure_history  = cuda::Buffer<float>(resource, cells * (configuration.pressure_iterations + 1u)),
                .projected_velocity = allocate_staggered_field(resource, configuration),
            },
            .extrapolation = {
                .valid_history   = cuda::Buffer<std::uint32_t>(resource, faces * (configuration.extrapolation_iterations + 1u)),
                .velocity_history = cuda::Buffer<float>(resource, faces * (configuration.extrapolation_iterations + 1u)),
                .velocity         = allocate_staggered_field(resource, configuration),
            },
            .advection = {
                .collision_masks      = cuda::Buffer<std::uint32_t>(resource, particle_count),
                .unconstrained_positions = allocate_vector_field(resource, particle_count),
            },
        };
        return cache;
    }

    DifferentialScratch allocate_differential_scratch(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration) {
        const std::size_t cells = cell_count(configuration);
        const std::size_t faces = total_face_count(configuration);
        return {
            .old_velocity_tangent                    = allocate_staggered_field(resource, configuration),
            .forced_velocity_tangent                 = allocate_staggered_field(resource, configuration),
            .divergence_tangent                      = cuda::Buffer<float>(resource, cells),
            .pressure_tangent_history                = cuda::Buffer<float>(resource, cells * (configuration.pressure_iterations + 1u)),
            .projected_velocity_tangent              = allocate_staggered_field(resource, configuration),
            .extrapolated_velocity_tangent_history   = cuda::Buffer<float>(resource, faces * (configuration.extrapolation_iterations + 1u)),
            .extrapolated_velocity_tangent           = allocate_staggered_field(resource, configuration),
            .old_velocity_adjoint                    = allocate_staggered_adjoint_field(resource, configuration),
            .forced_velocity_adjoint                 = allocate_staggered_adjoint_field(resource, configuration),
            .divergence_adjoint                      = cuda::Buffer<double>(resource, cells),
            .pressure_adjoint_history                = cuda::Buffer<double>(resource, cells * (configuration.pressure_iterations + 1u)),
            .projected_velocity_adjoint              = allocate_staggered_adjoint_field(resource, configuration),
            .extrapolated_velocity_adjoint_history   = cuda::Buffer<double>(resource, faces * (configuration.extrapolation_iterations + 1u)),
            .extrapolated_velocity_adjoint           = allocate_staggered_adjoint_field(resource, configuration),
        };
    }

    void zero(cuda::Resource& resource, VectorField& field) {
        resource.zero(field.x.data, field.x.size * sizeof(float));
        resource.zero(field.y.data, field.y.size * sizeof(float));
        resource.zero(field.z.data, field.z.size * sizeof(float));
    }

    void zero(cuda::Resource& resource, VectorAdjointField& field) {
        resource.zero(field.x.data, field.x.size * sizeof(double));
        resource.zero(field.y.data, field.y.size * sizeof(double));
        resource.zero(field.z.data, field.z.size * sizeof(double));
    }

    void zero(cuda::Resource& resource, MatrixField& field) {
        resource.zero(field.xx.data, field.xx.size * sizeof(float));
        resource.zero(field.xy.data, field.xy.size * sizeof(float));
        resource.zero(field.xz.data, field.xz.size * sizeof(float));
        resource.zero(field.yx.data, field.yx.size * sizeof(float));
        resource.zero(field.yy.data, field.yy.size * sizeof(float));
        resource.zero(field.yz.data, field.yz.size * sizeof(float));
        resource.zero(field.zx.data, field.zx.size * sizeof(float));
        resource.zero(field.zy.data, field.zy.size * sizeof(float));
        resource.zero(field.zz.data, field.zz.size * sizeof(float));
    }

    void zero(cuda::Resource& resource, MatrixAdjointField& field) {
        resource.zero(field.xx.data, field.xx.size * sizeof(double));
        resource.zero(field.xy.data, field.xy.size * sizeof(double));
        resource.zero(field.xz.data, field.xz.size * sizeof(double));
        resource.zero(field.yx.data, field.yx.size * sizeof(double));
        resource.zero(field.yy.data, field.yy.size * sizeof(double));
        resource.zero(field.yz.data, field.yz.size * sizeof(double));
        resource.zero(field.zx.data, field.zx.size * sizeof(double));
        resource.zero(field.zy.data, field.zy.size * sizeof(double));
        resource.zero(field.zz.data, field.zz.size * sizeof(double));
    }

    void zero(cuda::Resource& resource, StaggeredField& field) {
        zero_staggered(resource, field);
    }

    void zero(cuda::Resource& resource, StaggeredAdjointField& field) {
        zero_staggered(resource, field);
    }

    void copy(cuda::Resource& resource, const VectorField& source, VectorField& destination) {
        resource.copy_device(destination.x.data, source.x.data, source.x.size * sizeof(float));
        resource.copy_device(destination.y.data, source.y.data, source.y.size * sizeof(float));
        resource.copy_device(destination.z.data, source.z.data, source.z.size * sizeof(float));
    }

    void copy(cuda::Resource& resource, const VectorAdjointField& source, VectorAdjointField& destination) {
        resource.copy_device(destination.x.data, source.x.data, source.x.size * sizeof(double));
        resource.copy_device(destination.y.data, source.y.data, source.y.size * sizeof(double));
        resource.copy_device(destination.z.data, source.z.data, source.z.size * sizeof(double));
    }

    void copy(cuda::Resource& resource, const MatrixField& source, MatrixField& destination) {
        const std::array<const cuda::Buffer<float>*, 9u> sources{&source.xx, &source.xy, &source.xz, &source.yx, &source.yy, &source.yz, &source.zx, &source.zy, &source.zz};
        const std::array<cuda::Buffer<float>*, 9u> destinations{&destination.xx, &destination.xy, &destination.xz, &destination.yx, &destination.yy, &destination.yz, &destination.zx, &destination.zy, &destination.zz};
        for (std::size_t component = 0; component < 9u; ++component) resource.copy_device(destinations[component]->data, sources[component]->data, sources[component]->size * sizeof(float));
    }

    void copy(cuda::Resource& resource, const MatrixAdjointField& source, MatrixAdjointField& destination) {
        const std::array<const cuda::Buffer<double>*, 9u> sources{&source.xx, &source.xy, &source.xz, &source.yx, &source.yy, &source.yz, &source.zx, &source.zy, &source.zz};
        const std::array<cuda::Buffer<double>*, 9u> destinations{&destination.xx, &destination.xy, &destination.xz, &destination.yx, &destination.yy, &destination.yz, &destination.zx, &destination.zy, &destination.zz};
        for (std::size_t component = 0; component < 9u; ++component) resource.copy_device(destinations[component]->data, sources[component]->data, sources[component]->size * sizeof(double));
    }

    void copy(cuda::Resource& resource, const StaggeredField& source, StaggeredField& destination) {
        resource.copy_device(destination.x.data, source.x.data, source.x.size * sizeof(float));
        resource.copy_device(destination.y.data, source.y.data, source.y.size * sizeof(float));
        resource.copy_device(destination.z.data, source.z.data, source.z.size * sizeof(float));
    }

    void copy(cuda::Resource& resource, const StaggeredAdjointField& source, StaggeredAdjointField& destination) {
        resource.copy_device(destination.x.data, source.x.data, source.x.size * sizeof(double));
        resource.copy_device(destination.y.data, source.y.data, source.y.size * sizeof(double));
        resource.copy_device(destination.z.data, source.z.data, source.z.size * sizeof(double));
    }

    void accumulate(cuda::Resource& resource, const VectorAdjointField& source, VectorAdjointField& destination) {
        cuda_kernels::accumulate_vector_adjoint(static_cast<cudaStream_t>(resource.native_stream), source.x.size, view(source), view(destination));
    }

    void accumulate(cuda::Resource& resource, const MatrixAdjointField& source, MatrixAdjointField& destination) {
        const double* sources[]{source.xx.data, source.xy.data, source.xz.data, source.yx.data, source.yy.data, source.yz.data, source.zx.data, source.zy.data, source.zz.data};
        double* destinations[]{destination.xx.data, destination.xy.data, destination.xz.data, destination.yx.data, destination.yy.data, destination.yz.data, destination.zx.data, destination.zy.data, destination.zz.data};
        cuda_kernels::accumulate_matrix_adjoint(static_cast<cudaStream_t>(resource.native_stream), source.xx.size, sources, destinations);
    }

    void accumulate(cuda::Resource& resource, const StaggeredAdjointField& source, StaggeredAdjointField& destination) {
        cuda_kernels::accumulate_staggered_adjoint(static_cast<cudaStream_t>(resource.native_stream), source.x.size, source.y.size, source.z.size, view(source), view(destination));
    }

    void MarkerOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const TransferCache& transfer, ProjectionCache& cache) const {
        cuda_kernels::marker_forward(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(transfer.sorted_cell_keys.size), transfer.sorted_cell_keys.data, domain.cell_types.data, cache.cell_types.data);
    }

    void ParticleToGridOperator::forward(cuda::Resource& resource, const Configuration& configuration, const TransferKind kind, const VectorField& positions, const VectorField& velocities, const MatrixField* affine, const VectorField& external_accelerations, const cuda::Buffer<float>& masses, const Vector3 gravity, TransferCache& cache) const {
        const cudaStream_t stream            = static_cast<cudaStream_t>(resource.native_stream);
        const std::uint32_t particle_count   = static_cast<std::uint32_t>(positions.x.size);
        const std::size_t contribution_count = static_cast<std::size_t>(particle_count) * 81u;
        cuda_kernels::p2g_topology(stream, kernel_configuration(configuration), particle_count, view(positions), cache.unsorted_keys.data, cache.unsorted_ids.data);
        cuda_kernels::sort_contributions(stream, cache.sort_scratch.data, cache.sort_scratch.size, cache.unsorted_keys.data, cache.sorted_keys.data, cache.unsorted_ids.data, cache.sorted_ids.data, contribution_count);
        cuda_kernels::cell_topology(stream, kernel_configuration(configuration), particle_count, view(positions), cache.unsorted_cell_keys.data, cache.unsorted_cell_ids.data);
        cuda_kernels::sort_contributions(stream, cache.sort_scratch.data, cache.sort_scratch.size, cache.unsorted_cell_keys.data, cache.sorted_cell_keys.data, cache.unsorted_cell_ids.data, cache.sorted_cell_ids.data, particle_count);
        cuda_kernels::p2g_forward(stream, kernel_configuration(configuration), particle_count, kind == TransferKind::apic, view(positions), view(velocities), view(affine), view(external_accelerations), masses.data, gravity.x, gravity.y, gravity.z, cache.sorted_keys.data, cache.sorted_ids.data, cache.face_mass.data, view(cache.old_velocity), view(cache.forced_velocity));
    }

    void ParticleToGridOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const TransferKind kind, const VectorField& positions, const VectorField& velocities, const MatrixField* affine, const VectorField& external_accelerations, const cuda::Buffer<float>& masses, const Vector3 gravity, const VectorField& position_tangent, const VectorField& velocity_tangent, const MatrixField* affine_tangent, const VectorField& external_acceleration_tangent, const cuda::Buffer<float>& mass_tangent, const TransferCache& cache, DifferentialScratch& scratch) const {
        cuda_kernels::p2g_jvp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), kind == TransferKind::apic, view(positions), view(velocities), view(affine), view(external_accelerations), masses.data, gravity.x, gravity.y, gravity.z, view(position_tangent), view(velocity_tangent), view(affine_tangent), view(external_acceleration_tangent), mass_tangent.data, cache.sorted_keys.data, cache.sorted_ids.data, cache.face_mass.data, view(cache.old_velocity), view(cache.forced_velocity), view(scratch.old_velocity_tangent), view(scratch.forced_velocity_tangent));
    }

    void ParticleToGridOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const TransferKind kind, const VectorField& positions, const VectorField& velocities, const MatrixField* affine, const VectorField& external_accelerations, const cuda::Buffer<float>& masses, const Vector3 gravity, const TransferCache& cache, DifferentialScratch& scratch, VectorAdjointField& position_adjoint, VectorAdjointField& velocity_adjoint, MatrixAdjointField* affine_adjoint, VectorAdjointField& external_acceleration_adjoint, cuda::Buffer<double>& mass_adjoint) const {
        cuda_kernels::p2g_vjp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), kind == TransferKind::apic, view(positions), view(velocities), view(affine), view(external_accelerations), masses.data, gravity.x, gravity.y, gravity.z, cache.sorted_keys.data, cache.sorted_ids.data, cache.face_mass.data, view(cache.old_velocity), view(cache.forced_velocity), view(scratch.old_velocity_adjoint), view(scratch.forced_velocity_adjoint), view(position_adjoint), view(velocity_adjoint), view(affine_adjoint), view(external_acceleration_adjoint), mass_adjoint.data);
    }

    void ProjectionOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, ProjectionCache& cache, const StaggeredField& velocity) const {
        cuda_kernels::projection_forward(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), cache.cell_types.data, view(domain.solid_velocity), view(velocity), cache.divergence.data, cache.pressure_history.data, view(cache.projected_velocity));
    }

    void ProjectionOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain&, const ProjectionCache& cache, const StaggeredField& velocity_tangent, DifferentialScratch& scratch) const {
        cuda_kernels::projection_jvp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), cache.cell_types.data, view(velocity_tangent), scratch.divergence_tangent.data, scratch.pressure_tangent_history.data, view(scratch.projected_velocity_tangent));
    }

    void ProjectionOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain&, const ProjectionCache& cache, DifferentialScratch& scratch, StaggeredAdjointField& velocity_adjoint) const {
        cuda_kernels::projection_vjp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), cache.cell_types.data, view(scratch.projected_velocity_adjoint), scratch.divergence_adjoint.data, scratch.pressure_adjoint_history.data, view(velocity_adjoint));
    }

    void ExtrapolationOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ProjectionCache& projection, ExtrapolationCache& cache) const {
        cuda_kernels::extrapolation_forward(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), projection.cell_types.data, view(domain.solid_velocity), view(projection.projected_velocity), cache.valid_history.data, cache.velocity_history.data, view(cache.velocity));
    }

    void ExtrapolationOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain&, const ProjectionCache& projection, const ExtrapolationCache& cache, const StaggeredField& projected_velocity_tangent, DifferentialScratch& scratch) const {
        cuda_kernels::extrapolation_jvp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), projection.cell_types.data, cache.valid_history.data, view(projected_velocity_tangent), scratch.extrapolated_velocity_tangent_history.data, view(scratch.extrapolated_velocity_tangent));
    }

    void ExtrapolationOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain&, const ProjectionCache& projection, const ExtrapolationCache& cache, DifferentialScratch& scratch, StaggeredAdjointField& projected_velocity_adjoint) const {
        cuda_kernels::extrapolation_vjp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), projection.cell_types.data, cache.valid_history.data, view(scratch.extrapolated_velocity_adjoint), scratch.extrapolated_velocity_adjoint_history.data, view(projected_velocity_adjoint));
    }

    void GridToParticleOperator::forward_pic_flip(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const VectorField& old_particle_velocity, const TransferCache& transfer, const ExtrapolationCache& extrapolation, const cuda::Buffer<float>& blend, VectorField& velocity) const {
        cuda_kernels::g2p_pic_flip_forward(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), view(positions), view(old_particle_velocity), view(transfer.old_velocity), view(extrapolation.velocity), blend.data, view(velocity));
    }

    void GridToParticleOperator::jvp_pic_flip(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const VectorField& old_particle_velocity, const TransferCache& transfer, const ExtrapolationCache& extrapolation, const cuda::Buffer<float>& blend, const VectorField& position_tangent, const VectorField& old_particle_velocity_tangent, const cuda::Buffer<float>& blend_tangent, const DifferentialScratch& scratch, VectorField& velocity_tangent) const {
        cuda_kernels::g2p_pic_flip_jvp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), view(positions), view(old_particle_velocity), view(transfer.old_velocity), view(extrapolation.velocity), blend.data, view(position_tangent), view(old_particle_velocity_tangent), view(scratch.old_velocity_tangent), view(scratch.extrapolated_velocity_tangent), blend_tangent.data, view(velocity_tangent));
    }

    void GridToParticleOperator::vjp_pic_flip(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const VectorField& old_particle_velocity, const TransferCache& transfer, const ExtrapolationCache& extrapolation, const cuda::Buffer<float>& blend, const VectorAdjointField& velocity_adjoint, VectorAdjointField& position_adjoint, VectorAdjointField& old_particle_velocity_adjoint, StaggeredAdjointField& old_grid_velocity_adjoint, StaggeredAdjointField& extrapolated_velocity_adjoint, cuda::Buffer<double>& blend_adjoint) const {
        cuda_kernels::g2p_pic_flip_vjp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), view(positions), view(old_particle_velocity), view(transfer.old_velocity), view(extrapolation.velocity), transfer.sorted_keys.data, transfer.sorted_ids.data, blend.data, view(velocity_adjoint), view(position_adjoint), view(old_particle_velocity_adjoint), view(old_grid_velocity_adjoint), view(extrapolated_velocity_adjoint), transfer.reduction_values.data, transfer.sort_scratch.data, transfer.sort_scratch.size, transfer.reduction_result.data, blend_adjoint.data);
    }

    void GridToParticleOperator::forward_apic(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const TransferCache&, const ExtrapolationCache& extrapolation, VectorField& velocity, MatrixField& affine) const {
        const cuda_kernels::MatrixField matrix{.values = {affine.xx.data, affine.xy.data, affine.xz.data, affine.yx.data, affine.yy.data, affine.yz.data, affine.zx.data, affine.zy.data, affine.zz.data}};
        cuda_kernels::g2p_apic_forward(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), view(positions), view(extrapolation.velocity), view(velocity), matrix);
    }

    void GridToParticleOperator::jvp_apic(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const ExtrapolationCache& extrapolation, const VectorField& position_tangent, const StaggeredField& extrapolated_velocity_tangent, VectorField& velocity_tangent, MatrixField& affine_tangent) const {
        const cuda_kernels::MatrixField matrix{.values = {affine_tangent.xx.data, affine_tangent.xy.data, affine_tangent.xz.data, affine_tangent.yx.data, affine_tangent.yy.data, affine_tangent.yz.data, affine_tangent.zx.data, affine_tangent.zy.data, affine_tangent.zz.data}};
        cuda_kernels::g2p_apic_jvp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), view(positions), view(extrapolation.velocity), view(position_tangent), view(extrapolated_velocity_tangent), view(velocity_tangent), matrix);
    }

    void GridToParticleOperator::vjp_apic(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const TransferCache& transfer, const ExtrapolationCache& extrapolation, const VectorAdjointField& velocity_adjoint, const MatrixAdjointField& affine_adjoint, VectorAdjointField& position_adjoint, StaggeredAdjointField& extrapolated_velocity_adjoint) const {
        const cuda_kernels::ConstMatrixAdjointField matrix{.values = {affine_adjoint.xx.data, affine_adjoint.xy.data, affine_adjoint.xz.data, affine_adjoint.yx.data, affine_adjoint.yy.data, affine_adjoint.yz.data, affine_adjoint.zx.data, affine_adjoint.zy.data, affine_adjoint.zz.data}};
        cuda_kernels::g2p_apic_vjp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), view(positions), view(extrapolation.velocity), transfer.sorted_keys.data, transfer.sorted_ids.data, view(velocity_adjoint), matrix, view(position_adjoint), view(extrapolated_velocity_adjoint));
    }

    void ParticleAdvectionOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const float particle_radius, const VectorField& positions, const VectorField& velocities, AdvectionCache& cache, VectorField& next_positions, VectorField& next_velocities) const {
        cuda_kernels::advect_forward(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(positions.x.size), particle_radius, domain.cell_types.data, view(domain.solid_velocity), view(positions), view(velocities), cache.collision_masks.data, view(cache.unconstrained_positions), view(next_positions), view(next_velocities));
    }

    void ParticleAdvectionOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const float particle_radius, const AdvectionCache& cache, const VectorField& position_tangent, const VectorField& velocity_tangent, VectorField& next_position_tangent, VectorField& next_velocity_tangent) const {
        cuda_kernels::advect_jvp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(position_tangent.x.size), particle_radius, cache.collision_masks.data, view(position_tangent), view(velocity_tangent), view(next_position_tangent), view(next_velocity_tangent));
    }

    void ParticleAdvectionOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const float particle_radius, const AdvectionCache& cache, const VectorAdjointField& next_position_adjoint, const VectorAdjointField& next_velocity_adjoint, VectorAdjointField& position_adjoint, VectorAdjointField& velocity_adjoint) const {
        cuda_kernels::advect_vjp(static_cast<cudaStream_t>(resource.native_stream), kernel_configuration(configuration), static_cast<std::uint32_t>(next_position_adjoint.x.size), particle_radius, cache.collision_masks.data, view(next_position_adjoint), view(next_velocity_adjoint), view(position_adjoint), view(velocity_adjoint));
    }

} // namespace xayah::fluid::grid
