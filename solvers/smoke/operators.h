#ifndef XAYAH_SMOKE_OPERATORS_H
#define XAYAH_SMOKE_OPERATORS_H

#include <cstdint>
#include <cuda_runtime_api.h>

#include "cuda_types.h"

namespace xayah::smoke::cuda_kernels {
    struct VorticityView {
        CenteredVectorView centered_velocity;
        CenteredVectorView vorticity;
        ScalarView magnitude;
        CenteredVectorView normal;
        ScalarView normalizer;
    };

    struct ConstVorticityView {
        ConstCenteredVectorView centered_velocity;
        ConstCenteredVectorView vorticity;
        ConstScalarView magnitude;
        ConstCenteredVectorView normal;
        ConstScalarView normalizer;
    };

    struct VorticityTangentScratch {
        CenteredVectorView centered_velocity;
        CenteredVectorView vorticity;
        ScalarView magnitude;
        CenteredVectorView normal;
    };

    struct VorticityAdjointScratch {
        CenteredVectorAdjointView centered_velocity;
        CenteredVectorAdjointView vorticity;
        ScalarAdjointView magnitude;
        CenteredVectorAdjointView normal;
    };

    void accumulate(cudaStream_t stream, const double* source, double* destination, std::uint64_t count);

    void source_forward(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView state, ConstScalarView source, ScalarView output);
    void source_jvp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView state_tangent, ConstScalarView source_tangent, ScalarView output_tangent);
    void source_vjp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarAdjointView output_adjoint, ScalarAdjointView state_adjoint, ScalarAdjointView source_adjoint);

    void buoyancy_forward(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView density, ConstScalarView temperature, ConstCenteredVectorView external_acceleration, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, CenteredVectorView force);
    void buoyancy_jvp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView density, ConstScalarView temperature, ConstScalarView density_tangent, ConstScalarView temperature_tangent, ConstCenteredVectorView external_acceleration_tangent, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, const float* ambient_temperature_tangent, const float* density_buoyancy_tangent, const float* temperature_buoyancy_tangent, CenteredVectorView force_tangent);
    void buoyancy_vjp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView density, ConstScalarView temperature, const float* ambient_temperature, const float* density_buoyancy, const float* temperature_buoyancy, ConstCenteredVectorAdjointView force_adjoint, ScalarAdjointView density_adjoint, ScalarAdjointView temperature_adjoint, CenteredVectorAdjointView external_acceleration_adjoint, double* ambient_temperature_adjoint, double* density_buoyancy_adjoint, double* temperature_buoyancy_adjoint);

    void integrate_velocity_forward(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, ConstCenteredVectorView force, StaggeredVectorView output);
    void integrate_velocity_jvp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity_tangent, ConstCenteredVectorView force_tangent, StaggeredVectorView output_tangent);
    void integrate_velocity_vjp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView velocity_adjoint, CenteredVectorAdjointView force_adjoint);

    void advect_velocity_forward(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, StaggeredVectorView output);
    void advect_velocity_jvp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, ConstStaggeredVectorView velocity_tangent, VelocityBoundaryData boundary, StaggeredVectorView output_tangent);
    void advect_velocity_vjp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, ConstStaggeredVectorAdjointView output_adjoint, StaggeredVectorAdjointView velocity_adjoint);

    void constrain_velocity_forward(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView collider_velocity, ConstStaggeredVectorView velocity, VelocityBoundaryData boundary, StaggeredVectorView output);
    void constrain_velocity_jvp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity_tangent, VelocityBoundaryData boundary, StaggeredVectorView output_tangent);
    void constrain_velocity_vjp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorAdjointView output_adjoint, VelocityBoundaryData boundary, StaggeredVectorAdjointView velocity_adjoint);

    void advect_scalar_forward(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView collider_value, ConstScalarView source, ConstStaggeredVectorView velocity, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ScalarView output);
    void advect_scalar_jvp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView source, ConstScalarView source_tangent, ConstStaggeredVectorView velocity, ConstStaggeredVectorView velocity_tangent, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ScalarView output_tangent);
    void advect_scalar_vjp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstScalarView source, ConstStaggeredVectorView velocity, ScalarBoundaryData scalar_boundary, VelocityBoundaryData velocity_boundary, ConstScalarAdjointView output_adjoint, ScalarAdjointView source_adjoint, StaggeredVectorAdjointView velocity_adjoint);

    void vorticity_forward(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, const float* confinement, VorticityView cache, CenteredVectorView force);
    void vorticity_jvp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity_tangent, const float* confinement, const float* confinement_tangent, ConstVorticityView cache, CenteredVectorView force_tangent, VorticityTangentScratch tangent_scratch);
    void vorticity_vjp(cudaStream_t stream, Grid grid, const std::uint32_t* cell_mask, const float* confinement, ConstVorticityView cache, ConstCenteredVectorAdjointView force_adjoint, StaggeredVectorAdjointView velocity_adjoint, double* confinement_adjoint, VorticityAdjointScratch scratch);

    void projection_forward(cudaStream_t stream, Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity, VelocityBoundaryData velocity_boundary, ScalarBoundaryData pressure_boundary, ScalarView pressure, ScalarView rhs, StaggeredVectorView output);
    void projection_jvp(cudaStream_t stream, Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, ConstStaggeredVectorView velocity_tangent, VelocityBoundaryData velocity_boundary, ScalarBoundaryData pressure_boundary, ScalarView pressure_tangent, ScalarView rhs_tangent, StaggeredVectorView output_tangent);
    void projection_vjp(cudaStream_t stream, Grid grid, std::uint32_t iterations, std::uint32_t pressure_anchor, const std::uint32_t* cell_mask, ConstStaggeredVectorAdjointView output_adjoint, VelocityBoundaryData velocity_boundary, ScalarBoundaryData pressure_boundary, ScalarAdjointView pressure_adjoint, ScalarAdjointView rhs_adjoint, StaggeredVectorAdjointView velocity_adjoint);

} // namespace xayah::smoke::cuda_kernels

#endif
