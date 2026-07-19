module;

#include "operators.h"

#include <cuda_runtime_api.h>

module xayah.smoke.operators;

import xayah.cuda;
import xayah.smoke.data;

namespace xayah::smoke {

    namespace {

        cuda_kernels::Grid grid(const Configuration& configuration) {
            return {.nx = static_cast<int>(configuration.resolution[0]), .ny = static_cast<int>(configuration.resolution[1]), .nz = static_cast<int>(configuration.resolution[2]), .cell_size = configuration.cell_size, .time_step = configuration.time_step};
        }

        cuda_kernels::ConstScalarView scalar(const ScalarField& field) {
            return {.values = field.values.data};
        }

        cuda_kernels::ScalarView scalar(ScalarField& field) {
            return {.values = field.values.data};
        }

        cuda_kernels::ConstCenteredVectorView centered(const CenteredVectorField& field) {
            return {.x = field.x.values.data, .y = field.y.values.data, .z = field.z.values.data};
        }

        cuda_kernels::CenteredVectorView centered(CenteredVectorField& field) {
            return {.x = field.x.values.data, .y = field.y.values.data, .z = field.z.values.data};
        }

        cuda_kernels::ConstStaggeredVectorView staggered(const StaggeredVectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::StaggeredVectorView staggered(StaggeredVectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::ConstScalarAdjointView scalar(const ScalarAdjointField& field) {
            return {.values = field.values.data};
        }

        cuda_kernels::ScalarAdjointView scalar(ScalarAdjointField& field) {
            return {.values = field.values.data};
        }

        cuda_kernels::ConstCenteredVectorAdjointView centered(const CenteredVectorAdjointField& field) {
            return {.x = field.x.values.data, .y = field.y.values.data, .z = field.z.values.data};
        }

        cuda_kernels::CenteredVectorAdjointView centered(CenteredVectorAdjointField& field) {
            return {.x = field.x.values.data, .y = field.y.values.data, .z = field.z.values.data};
        }

        cuda_kernels::ConstStaggeredVectorAdjointView staggered(const StaggeredVectorAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::StaggeredVectorAdjointView staggered(StaggeredVectorAdjointField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        cuda_kernels::VorticityView vorticity(VorticityCache& cache) {
            return {.centered_velocity = centered(cache.centered_velocity), .vorticity = centered(cache.vorticity), .magnitude = scalar(cache.magnitude), .normal = centered(cache.normal), .normalizer = scalar(cache.normalizer)};
        }

        cuda_kernels::ConstVorticityView vorticity(const VorticityCache& cache) {
            return {.centered_velocity = centered(cache.centered_velocity), .vorticity = centered(cache.vorticity), .magnitude = scalar(cache.magnitude), .normal = centered(cache.normal), .normalizer = scalar(cache.normalizer)};
        }

        cuda_kernels::VorticityTangentScratch vorticity_scratch(VorticityCache& cache) {
            return {.centered_velocity = centered(cache.centered_velocity), .vorticity = centered(cache.vorticity), .magnitude = scalar(cache.magnitude), .normal = centered(cache.normal)};
        }

        cuda_kernels::VorticityAdjointScratch vorticity_scratch(VorticityAdjointCache& cache) {
            return {.centered_velocity = centered(cache.centered_velocity), .vorticity = centered(cache.vorticity), .magnitude = scalar(cache.magnitude), .normal = centered(cache.normal)};
        }

        cuda_kernels::ScalarBoundaryData scalar_boundary(const ScalarBoundary& boundary) {
            const std::array faces{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max};
            cuda_kernels::ScalarBoundaryData packed{};
            for (std::size_t face = 0u; face < faces.size(); ++face) {
                packed.modes[face]  = static_cast<std::uint32_t>(faces[face].mode);
                packed.values[face] = faces[face].value;
            }
            return packed;
        }

        cuda_kernels::VelocityBoundaryData velocity_boundary(const VelocityBoundary& boundary) {
            const std::array faces{boundary.x_min, boundary.x_max, boundary.y_min, boundary.y_max, boundary.z_min, boundary.z_max};
            cuda_kernels::VelocityBoundaryData packed{};
            for (std::size_t face = 0u; face < faces.size(); ++face) {
                packed.modes[face]            = static_cast<std::uint32_t>(faces[face].mode);
                packed.values[face * 3u]      = faces[face].value.x;
                packed.values[face * 3u + 1u] = faces[face].value.y;
                packed.values[face * 3u + 2u] = faces[face].value.z;
            }
            return packed;
        }

        cudaStream_t stream(cuda::Resource& resource) {
            return static_cast<cudaStream_t>(resource.native_stream);
        }

    } // namespace

    void SourceOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& state, const ScalarField& source, ScalarField& output) const {
        cuda_kernels::source_forward(stream(resource), grid(configuration), domain.cell_mask.data, scalar(state), scalar(source), scalar(output));
    }

    void SourceOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& state_tangent, const ScalarField& source_tangent, ScalarField& output_tangent) const {
        cuda_kernels::source_jvp(stream(resource), grid(configuration), domain.cell_mask.data, scalar(state_tangent), scalar(source_tangent), scalar(output_tangent));
    }

    void SourceOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarAdjointField& output_adjoint, ScalarAdjointField& state_adjoint, ScalarAdjointField& source_adjoint) const {
        cuda_kernels::source_vjp(stream(resource), grid(configuration), domain.cell_mask.data, scalar(output_adjoint), scalar(state_adjoint), scalar(source_adjoint));
    }

    void ForceAssemblyOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& density, const ScalarField& temperature, const StaggeredVectorField& velocity, const Control& control, const Parameters& parameters, CenteredVectorField& force, VorticityCache& vorticity_cache) const {
        cuda_kernels::buoyancy_forward(stream(resource), grid(configuration), domain.cell_mask.data, scalar(density), scalar(temperature), centered(control.external_acceleration), parameters.ambient_temperature.data, parameters.density_buoyancy.data, parameters.temperature_buoyancy.data, centered(force));
        if (configuration.vorticity_confinement_enabled) cuda_kernels::vorticity_forward(stream(resource), grid(configuration), domain.cell_mask.data, staggered(velocity), parameters.vorticity_confinement.data, vorticity(vorticity_cache), centered(force));
    }

    void ForceAssemblyOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& density, const ScalarField& temperature, const StaggeredVectorField& velocity_tangent, const ScalarField& density_tangent, const ScalarField& temperature_tangent, const ControlTangent& control_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, CenteredVectorField& force_tangent, const VorticityCache& vorticity_cache, VorticityCache& scratch) const {
        cuda_kernels::buoyancy_jvp(stream(resource), grid(configuration), domain.cell_mask.data, scalar(density), scalar(temperature), scalar(density_tangent), scalar(temperature_tangent), centered(control_tangent.external_acceleration), parameters.ambient_temperature.data, parameters.density_buoyancy.data, parameters.temperature_buoyancy.data, parameter_tangent.ambient_temperature.data, parameter_tangent.density_buoyancy.data, parameter_tangent.temperature_buoyancy.data, centered(force_tangent));
        if (configuration.vorticity_confinement_enabled) cuda_kernels::vorticity_jvp(stream(resource), grid(configuration), domain.cell_mask.data, staggered(velocity_tangent), parameters.vorticity_confinement.data, parameter_tangent.vorticity_confinement.data, vorticity(vorticity_cache), centered(force_tangent), vorticity_scratch(scratch));
    }

    void ForceAssemblyOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& density, const ScalarField& temperature, const StaggeredVectorField& velocity, const Parameters& parameters, const CenteredVectorAdjointField& force_adjoint, const VorticityCache& vorticity_cache, ScalarAdjointField& density_adjoint, ScalarAdjointField& temperature_adjoint, StaggeredVectorAdjointField& velocity_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, VorticityAdjointCache& scratch) const {
        cuda_kernels::buoyancy_vjp(stream(resource), grid(configuration), domain.cell_mask.data, scalar(density), scalar(temperature), parameters.ambient_temperature.data, parameters.density_buoyancy.data, parameters.temperature_buoyancy.data, centered(force_adjoint), scalar(density_adjoint), scalar(temperature_adjoint), centered(control_adjoint.external_acceleration), parameter_adjoint.ambient_temperature.data, parameter_adjoint.density_buoyancy.data, parameter_adjoint.temperature_buoyancy.data);
        if (configuration.vorticity_confinement_enabled) cuda_kernels::vorticity_vjp(stream(resource), grid(configuration), domain.cell_mask.data, parameters.vorticity_confinement.data, vorticity(vorticity_cache), centered(force_adjoint), staggered(velocity_adjoint), parameter_adjoint.vorticity_confinement.data, vorticity_scratch(scratch));
    }

    void VelocityEvolutionOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& velocity, const CenteredVectorField& force, StaggeredVectorField& forced_velocity, StaggeredVectorField& raw_advected_velocity, StaggeredVectorField& advected_velocity) const {
        cuda_kernels::integrate_velocity_forward(stream(resource), grid(configuration), domain.cell_mask.data, staggered(velocity), centered(force), staggered(forced_velocity));
        cuda_kernels::advect_velocity_forward(stream(resource), grid(configuration), domain.cell_mask.data, staggered(forced_velocity), velocity_boundary(configuration.velocity_boundary), staggered(raw_advected_velocity));
        cuda_kernels::constrain_velocity_forward(stream(resource), grid(configuration), domain.cell_mask.data, staggered(domain.collider_velocity), staggered(raw_advected_velocity), velocity_boundary(configuration.velocity_boundary), staggered(advected_velocity));
    }

    void VelocityEvolutionOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField&, const StaggeredVectorField& velocity_tangent, const CenteredVectorField& force_tangent, const StaggeredVectorField& forced_velocity, StaggeredVectorField& forced_velocity_tangent, StaggeredVectorField& raw_advected_velocity_tangent, StaggeredVectorField& advected_velocity_tangent) const {
        cuda_kernels::integrate_velocity_jvp(stream(resource), grid(configuration), domain.cell_mask.data, staggered(velocity_tangent), centered(force_tangent), staggered(forced_velocity_tangent));
        cuda_kernels::advect_velocity_jvp(stream(resource), grid(configuration), domain.cell_mask.data, staggered(forced_velocity), staggered(forced_velocity_tangent), velocity_boundary(configuration.velocity_boundary), staggered(raw_advected_velocity_tangent));
        cuda_kernels::constrain_velocity_jvp(stream(resource), grid(configuration), domain.cell_mask.data, staggered(raw_advected_velocity_tangent), velocity_boundary(configuration.velocity_boundary), staggered(advected_velocity_tangent));
    }

    void VelocityEvolutionOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& forced_velocity, const StaggeredVectorAdjointField& advected_velocity_adjoint, StaggeredVectorAdjointField& raw_advected_velocity_adjoint, StaggeredVectorAdjointField& forced_velocity_adjoint, StaggeredVectorAdjointField& velocity_adjoint, CenteredVectorAdjointField& force_adjoint) const {
        cuda_kernels::constrain_velocity_vjp(stream(resource), grid(configuration), domain.cell_mask.data, staggered(advected_velocity_adjoint), velocity_boundary(configuration.velocity_boundary), staggered(raw_advected_velocity_adjoint));
        cuda_kernels::advect_velocity_vjp(stream(resource), grid(configuration), domain.cell_mask.data, staggered(forced_velocity), velocity_boundary(configuration.velocity_boundary), staggered(raw_advected_velocity_adjoint), staggered(forced_velocity_adjoint));
        cuda_kernels::integrate_velocity_vjp(stream(resource), grid(configuration), domain.cell_mask.data, staggered(forced_velocity_adjoint), staggered(velocity_adjoint), centered(force_adjoint));
    }

    void PressureProjectionOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& velocity, ScalarField& pressure, ScalarField& rhs, StaggeredVectorField& projected_velocity) const {
        cuda_kernels::projection_forward(stream(resource), grid(configuration), configuration.pressure_iterations, domain.pressure_anchor, domain.cell_mask.data, staggered(velocity), velocity_boundary(configuration.velocity_boundary), scalar_boundary(configuration.pressure_boundary), scalar(pressure), scalar(rhs), staggered(projected_velocity));
    }

    void PressureProjectionOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& velocity_tangent, ScalarField& pressure_tangent, ScalarField& rhs_tangent, StaggeredVectorField& projected_velocity_tangent) const {
        cuda_kernels::projection_jvp(stream(resource), grid(configuration), configuration.pressure_iterations, domain.pressure_anchor, domain.cell_mask.data, staggered(velocity_tangent), velocity_boundary(configuration.velocity_boundary), scalar_boundary(configuration.pressure_boundary), scalar(pressure_tangent), scalar(rhs_tangent), staggered(projected_velocity_tangent));
    }

    void PressureProjectionOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorAdjointField& projected_velocity_adjoint, ScalarAdjointField& pressure_adjoint, ScalarAdjointField& rhs_adjoint, StaggeredVectorAdjointField& velocity_adjoint) const {
        cuda_kernels::projection_vjp(stream(resource), grid(configuration), configuration.pressure_iterations, domain.pressure_anchor, domain.cell_mask.data, staggered(projected_velocity_adjoint), velocity_boundary(configuration.velocity_boundary), scalar_boundary(configuration.pressure_boundary), scalar(pressure_adjoint), scalar(rhs_adjoint), staggered(velocity_adjoint));
    }

    void ScalarAdvectionOperator::forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarBoundary& boundary, const ScalarField& collider_value, const ScalarField& source, const StaggeredVectorField& velocity, ScalarField& output) const {
        cuda_kernels::advect_scalar_forward(stream(resource), grid(configuration), domain.cell_mask.data, scalar(collider_value), scalar(source), staggered(velocity), scalar_boundary(boundary), velocity_boundary(configuration.velocity_boundary), scalar(output));
    }

    void ScalarAdvectionOperator::jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarBoundary& boundary, const ScalarField& source, const ScalarField& source_tangent, const StaggeredVectorField& velocity, const StaggeredVectorField& velocity_tangent, ScalarField& output_tangent) const {
        cuda_kernels::advect_scalar_jvp(stream(resource), grid(configuration), domain.cell_mask.data, scalar(source), scalar(source_tangent), staggered(velocity), staggered(velocity_tangent), scalar_boundary(boundary), velocity_boundary(configuration.velocity_boundary), scalar(output_tangent));
    }

    void ScalarAdvectionOperator::vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarBoundary& boundary, const ScalarField& source, const StaggeredVectorField& velocity, const ScalarAdjointField& output_adjoint, ScalarAdjointField& source_adjoint, StaggeredVectorAdjointField& velocity_adjoint) const {
        cuda_kernels::advect_scalar_vjp(stream(resource), grid(configuration), domain.cell_mask.data, scalar(source), staggered(velocity), scalar_boundary(boundary), velocity_boundary(configuration.velocity_boundary), scalar(output_adjoint), scalar(source_adjoint), staggered(velocity_adjoint));
    }

} // namespace xayah::smoke
