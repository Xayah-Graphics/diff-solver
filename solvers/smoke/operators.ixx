export module xayah.smoke.operators;

import xayah.cuda;
import xayah.smoke.data;

export namespace xayah::smoke {

    struct SourceOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& state, const ScalarField& source, ScalarField& output) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& state_tangent, const ScalarField& source_tangent, ScalarField& output_tangent) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarAdjointField& output_adjoint, ScalarAdjointField& state_adjoint, ScalarAdjointField& source_adjoint) const;
    };

    struct ForceAssemblyOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& density, const ScalarField& temperature, const StaggeredVectorField& velocity, const Control& control, const Parameters& parameters, CenteredVectorField& force, VorticityCache& vorticity_cache) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& density, const ScalarField& temperature, const StaggeredVectorField& velocity_tangent, const ScalarField& density_tangent, const ScalarField& temperature_tangent, const ControlTangent& control_tangent, const Parameters& parameters, const ParameterTangent& parameter_tangent, CenteredVectorField& force_tangent, const VorticityCache& vorticity_cache, VorticityCache& scratch) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarField& density, const ScalarField& temperature, const StaggeredVectorField& velocity, const Parameters& parameters, const CenteredVectorAdjointField& force_adjoint, const VorticityCache& vorticity_cache, ScalarAdjointField& density_adjoint, ScalarAdjointField& temperature_adjoint, StaggeredVectorAdjointField& velocity_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, VorticityAdjointCache& scratch) const;
    };

    struct VelocityEvolutionOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& velocity, const CenteredVectorField& force, StaggeredVectorField& forced_velocity, StaggeredVectorField& raw_advected_velocity, StaggeredVectorField& advected_velocity) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& velocity, const StaggeredVectorField& velocity_tangent, const CenteredVectorField& force_tangent, const StaggeredVectorField& forced_velocity, StaggeredVectorField& forced_velocity_tangent, StaggeredVectorField& raw_advected_velocity_tangent, StaggeredVectorField& advected_velocity_tangent) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& forced_velocity, const StaggeredVectorAdjointField& advected_velocity_adjoint, StaggeredVectorAdjointField& raw_advected_velocity_adjoint, StaggeredVectorAdjointField& forced_velocity_adjoint, StaggeredVectorAdjointField& velocity_adjoint, CenteredVectorAdjointField& force_adjoint) const;
    };

    struct PressureProjectionOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& velocity, ScalarField& pressure, ScalarField& rhs, StaggeredVectorField& projected_velocity) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorField& velocity_tangent, ScalarField& pressure_tangent, ScalarField& rhs_tangent, StaggeredVectorField& projected_velocity_tangent) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const StaggeredVectorAdjointField& projected_velocity_adjoint, ScalarAdjointField& pressure_adjoint, ScalarAdjointField& rhs_adjoint, StaggeredVectorAdjointField& velocity_adjoint) const;
    };

    struct ScalarAdvectionOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarBoundary& boundary, const ScalarField& collider_value, const ScalarField& source, const StaggeredVectorField& velocity, ScalarField& output) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarBoundary& boundary, const ScalarField& source, const ScalarField& source_tangent, const StaggeredVectorField& velocity, const StaggeredVectorField& velocity_tangent, ScalarField& output_tangent) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ScalarBoundary& boundary, const ScalarField& source, const StaggeredVectorField& velocity, const ScalarAdjointField& output_adjoint, ScalarAdjointField& source_adjoint, StaggeredVectorAdjointField& velocity_adjoint) const;
    };

} // namespace xayah::smoke
