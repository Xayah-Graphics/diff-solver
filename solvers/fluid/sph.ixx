export module xayah.fluid.sph;

import std;
import xayah.cuda;
import xayah.fluid.data;

export namespace xayah::fluid::sph {

    struct DeviceBoundary {
        VectorField positions;
        VectorField velocities;
        ScalarField volumes;
    };

    [[nodiscard]] VectorField allocate_vector_field(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] VectorAdjointField allocate_vector_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] ScalarField allocate_scalar_field(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] ScalarAdjointField allocate_scalar_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] ParticleParameters allocate_particle_parameters(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] ParticleParameterTangent allocate_particle_parameter_tangent(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] ParticleParameterAdjoint allocate_particle_parameter_adjoint(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] State allocate_state(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] Control allocate_control(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] StateTangent allocate_state_tangent(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] ControlTangent allocate_control_tangent(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] StateAdjoint allocate_state_adjoint(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] ControlAdjoint allocate_control_adjoint(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] DeviceBoundary allocate_boundary(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration);
    [[nodiscard]] NeighborSearch allocate_neighbor_search(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration);
    [[nodiscard]] Neighborhood allocate_neighborhood(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration);

    void copy_state(cuda::Resource& resource, const State& source, State& destination);
    void copy_state_tangent(cuda::Resource& resource, const StateTangent& source, StateTangent& destination);
    void copy_state_adjoint(cuda::Resource& resource, const StateAdjoint& source, StateAdjoint& destination);
    void accumulate_state_adjoint(cuda::Resource& resource, const StateAdjoint& source, StateAdjoint& destination);
    void copy_vector(cuda::Resource& resource, const VectorField& source, VectorField& destination);
    void copy_scalar(cuda::Resource& resource, const ScalarField& source, ScalarField& destination);
    void copy_vector_adjoint(cuda::Resource& resource, const VectorAdjointField& source, VectorAdjointField& destination);
    void copy_scalar_adjoint(cuda::Resource& resource, const ScalarAdjointField& source, ScalarAdjointField& destination);
    void accumulate_vector_adjoint(cuda::Resource& resource, const VectorAdjointField& source, VectorAdjointField& destination);
    void accumulate_scalar_adjoint(cuda::Resource& resource, const ScalarAdjointField& source, ScalarAdjointField& destination);
    void zero_vector(cuda::Resource& resource, VectorField& field);
    void zero_vector_adjoint(cuda::Resource& resource, VectorAdjointField& field);
    void zero_scalar(cuda::Resource& resource, ScalarField& field);
    void zero_scalar_adjoint(cuda::Resource& resource, ScalarAdjointField& field);

    void build_neighborhood(cuda::Resource& resource, const Configuration& configuration, std::uint64_t step_index, const DeviceBoundary& boundary, const VectorField& positions, NeighborSearch& search, Neighborhood& neighborhood);
    void density_forward(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities);
    void density_jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent);
    void density_vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint);
    void density_forward_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities);
    void density_jvp_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent);
    void density_vjp_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint);
    void pbf_density_forward(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities);
    void pbf_density_jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent);
    void pbf_density_vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint);
    void pbf_density_forward_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, ScalarField& densities);
    void pbf_density_jvp_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const VectorField& position_tangent, const ParticleParameters& parameters, const ParticleParameterTangent& parameter_tangent, const Neighborhood& neighborhood, ScalarField& density_tangent);
    void pbf_density_vjp_frozen(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& topology_positions, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarAdjointField& density_adjoint, VectorAdjointField& position_adjoint, ParticleParameterAdjoint& parameter_adjoint);

    void non_pressure_forward(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const State& state, const Control& control, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, VectorField& accelerations);
    void non_pressure_jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const State& state, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParticleParameterTangent& parameter_tangent, const ScalarField& density_tangent, VectorField& acceleration_tangent);
    void non_pressure_vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const State& state, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const VectorAdjointField& acceleration_adjoint, StateAdjoint& state_adjoint, ControlAdjoint& control_adjoint, ScalarAdjointField& density_adjoint, ParticleParameterAdjoint& parameter_adjoint);

    void pressure_acceleration_forward(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, VectorField& accelerations);
    void pressure_acceleration_jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, const VectorField& position_tangent, const ParticleParameterTangent& parameter_tangent, const ScalarField& density_tangent, const ScalarField& pressure_tangent, VectorField& acceleration_tangent);
    void pressure_acceleration_vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceBoundary& boundary, const VectorField& positions, const ParticleParameters& parameters, const Neighborhood& neighborhood, const ScalarField& densities, const ScalarField& pressures, const VectorAdjointField& acceleration_adjoint, VectorAdjointField& position_adjoint, ScalarAdjointField& density_adjoint, ScalarAdjointField& pressure_adjoint, ParticleParameterAdjoint& parameter_adjoint);

    void add_accelerations(cuda::Resource& resource, const VectorField& first, const VectorField& second, VectorField& output);
    void integrate_forward(cuda::Resource& resource, const Configuration& configuration, const State& state, const VectorField& accelerations, State& next_state);
    void integrate_jvp(cuda::Resource& resource, const Configuration& configuration, const State& state, const VectorField& accelerations, const StateTangent& state_tangent, const VectorField& acceleration_tangent, StateTangent& next_state_tangent);
    void integrate_vjp(cuda::Resource& resource, const Configuration& configuration, const State& state, const VectorField& accelerations, const StateAdjoint& next_state_adjoint, StateAdjoint& state_adjoint, VectorAdjointField& acceleration_adjoint);

} // namespace xayah::fluid::sph
