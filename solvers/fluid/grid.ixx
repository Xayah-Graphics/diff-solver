export module xayah.fluid.grid;

import std;
import xayah.cuda;
import xayah.fluid.data;

export namespace xayah::fluid::grid {

    enum class TransferKind : std::uint32_t {
        pic_flip,
        apic,
    };

    enum class CellType : std::uint32_t {
        air,
        fluid,
        solid,
    };

    struct Configuration {
        std::array<std::uint32_t, 3u> resolution;
        Vector3 origin;
        float cell_size;
        float time_step;
        std::uint32_t pressure_iterations;
        std::uint32_t extrapolation_iterations;
        std::vector<std::uint32_t> solid_cell_mask;
        std::array<std::vector<float>, 3u> solid_face_velocities;
        std::optional<std::uint32_t> pressure_anchor;
    };

    struct StaggeredField {
        cuda::Buffer<float> x;
        cuda::Buffer<float> y;
        cuda::Buffer<float> z;
    };

    struct StaggeredAdjointField {
        cuda::Buffer<double> x;
        cuda::Buffer<double> y;
        cuda::Buffer<double> z;
    };

    struct DeviceDomain {
        cuda::Buffer<std::uint32_t> cell_types;
        StaggeredField solid_velocity;
    };

    struct TransferCache {
        cuda::Buffer<std::uint64_t> unsorted_keys;
        cuda::Buffer<std::uint64_t> sorted_keys;
        cuda::Buffer<std::uint32_t> unsorted_ids;
        cuda::Buffer<std::uint32_t> sorted_ids;
        cuda::Buffer<std::uint64_t> unsorted_cell_keys;
        cuda::Buffer<std::uint64_t> sorted_cell_keys;
        cuda::Buffer<std::uint32_t> unsorted_cell_ids;
        cuda::Buffer<std::uint32_t> sorted_cell_ids;
        cuda::Buffer<float> face_mass;
        StaggeredField old_velocity;
        StaggeredField forced_velocity;
        cuda::Buffer<std::byte> sort_scratch;
        cuda::Buffer<double> reduction_values;
        cuda::Buffer<double> reduction_result;
    };

    struct ProjectionCache {
        cuda::Buffer<std::uint32_t> cell_types;
        cuda::Buffer<float> divergence;
        cuda::Buffer<float> pressure_history;
        StaggeredField projected_velocity;
    };

    struct ExtrapolationCache {
        cuda::Buffer<std::uint32_t> valid_history;
        cuda::Buffer<float> velocity_history;
        StaggeredField velocity;
    };

    struct AdvectionCache {
        cuda::Buffer<std::uint32_t> collision_masks;
        VectorField unconstrained_positions;
    };

    struct StepCache {
        TransferCache transfer;
        ProjectionCache projection;
        ExtrapolationCache extrapolation;
        AdvectionCache advection;
    };

    struct DifferentialScratch {
        StaggeredField old_velocity_tangent;
        StaggeredField forced_velocity_tangent;
        cuda::Buffer<float> divergence_tangent;
        cuda::Buffer<float> pressure_tangent_history;
        StaggeredField projected_velocity_tangent;
        cuda::Buffer<float> extrapolated_velocity_tangent_history;
        StaggeredField extrapolated_velocity_tangent;
        StaggeredAdjointField old_velocity_adjoint;
        StaggeredAdjointField forced_velocity_adjoint;
        cuda::Buffer<double> divergence_adjoint;
        cuda::Buffer<double> pressure_adjoint_history;
        StaggeredAdjointField projected_velocity_adjoint;
        cuda::Buffer<double> extrapolated_velocity_adjoint_history;
        StaggeredAdjointField extrapolated_velocity_adjoint;
    };

    [[nodiscard]] std::array<std::size_t, 3u> face_counts(const Configuration& configuration);
    [[nodiscard]] std::size_t cell_count(const Configuration& configuration);
    [[nodiscard]] std::size_t total_face_count(const Configuration& configuration);
    [[nodiscard]] DeviceDomain allocate_domain(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration);
    [[nodiscard]] VectorField allocate_vector_field(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] VectorAdjointField allocate_vector_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] MatrixField allocate_matrix_field(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] MatrixAdjointField allocate_matrix_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, std::size_t count);
    [[nodiscard]] StaggeredField allocate_staggered_field(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration);
    [[nodiscard]] StaggeredAdjointField allocate_staggered_adjoint_field(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration);
    [[nodiscard]] StepCache allocate_step_cache(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration, std::uint32_t particle_count);
    [[nodiscard]] DifferentialScratch allocate_differential_scratch(const std::shared_ptr<cuda::Resource>& resource, const Configuration& configuration);
    void zero(cuda::Resource& resource, VectorField& field);
    void zero(cuda::Resource& resource, VectorAdjointField& field);
    void zero(cuda::Resource& resource, MatrixField& field);
    void zero(cuda::Resource& resource, MatrixAdjointField& field);
    void zero(cuda::Resource& resource, StaggeredField& field);
    void zero(cuda::Resource& resource, StaggeredAdjointField& field);
    void copy(cuda::Resource& resource, const VectorField& source, VectorField& destination);
    void copy(cuda::Resource& resource, const VectorAdjointField& source, VectorAdjointField& destination);
    void copy(cuda::Resource& resource, const MatrixField& source, MatrixField& destination);
    void copy(cuda::Resource& resource, const MatrixAdjointField& source, MatrixAdjointField& destination);
    void copy(cuda::Resource& resource, const StaggeredField& source, StaggeredField& destination);
    void copy(cuda::Resource& resource, const StaggeredAdjointField& source, StaggeredAdjointField& destination);
    void accumulate(cuda::Resource& resource, const VectorAdjointField& source, VectorAdjointField& destination);
    void accumulate(cuda::Resource& resource, const MatrixAdjointField& source, MatrixAdjointField& destination);
    void accumulate(cuda::Resource& resource, const StaggeredAdjointField& source, StaggeredAdjointField& destination);

    struct MarkerOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const TransferCache& transfer, ProjectionCache& cache) const;
    };

    struct ParticleToGridOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, TransferKind kind, const VectorField& positions, const VectorField& velocities, const MatrixField* affine, const VectorField& external_accelerations, const cuda::Buffer<float>& masses, Vector3 gravity, TransferCache& cache) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, TransferKind kind, const VectorField& positions, const VectorField& velocities, const MatrixField* affine, const VectorField& external_accelerations, const cuda::Buffer<float>& masses, Vector3 gravity, const VectorField& position_tangent, const VectorField& velocity_tangent, const MatrixField* affine_tangent, const VectorField& external_acceleration_tangent, const cuda::Buffer<float>& mass_tangent, const TransferCache& cache, DifferentialScratch& scratch) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, TransferKind kind, const VectorField& positions, const VectorField& velocities, const MatrixField* affine, const VectorField& external_accelerations, const cuda::Buffer<float>& masses, Vector3 gravity, const TransferCache& cache, DifferentialScratch& scratch, VectorAdjointField& position_adjoint, VectorAdjointField& velocity_adjoint, MatrixAdjointField* affine_adjoint, VectorAdjointField& external_acceleration_adjoint, cuda::Buffer<double>& mass_adjoint) const;
    };

    struct ProjectionOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, ProjectionCache& cache, const StaggeredField& velocity) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ProjectionCache& cache, const StaggeredField& velocity_tangent, DifferentialScratch& scratch) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ProjectionCache& cache, DifferentialScratch& scratch, StaggeredAdjointField& velocity_adjoint) const;
    };

    struct ExtrapolationOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ProjectionCache& projection, ExtrapolationCache& cache) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ProjectionCache& projection, const ExtrapolationCache& cache, const StaggeredField& projected_velocity_tangent, DifferentialScratch& scratch) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, const ProjectionCache& projection, const ExtrapolationCache& cache, DifferentialScratch& scratch, StaggeredAdjointField& projected_velocity_adjoint) const;
    };

    struct GridToParticleOperator {
        void forward_pic_flip(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const VectorField& old_particle_velocity, const TransferCache& transfer, const ExtrapolationCache& extrapolation, const cuda::Buffer<float>& blend, VectorField& velocity) const;
        void jvp_pic_flip(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const VectorField& old_particle_velocity, const TransferCache& transfer, const ExtrapolationCache& extrapolation, const cuda::Buffer<float>& blend, const VectorField& position_tangent, const VectorField& old_particle_velocity_tangent, const cuda::Buffer<float>& blend_tangent, const DifferentialScratch& scratch, VectorField& velocity_tangent) const;
        void vjp_pic_flip(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const VectorField& old_particle_velocity, const TransferCache& transfer, const ExtrapolationCache& extrapolation, const cuda::Buffer<float>& blend, const VectorAdjointField& velocity_adjoint, VectorAdjointField& position_adjoint, VectorAdjointField& old_particle_velocity_adjoint, StaggeredAdjointField& old_grid_velocity_adjoint, StaggeredAdjointField& extrapolated_velocity_adjoint, cuda::Buffer<double>& blend_adjoint) const;
        void forward_apic(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const TransferCache& transfer, const ExtrapolationCache& extrapolation, VectorField& velocity, MatrixField& affine) const;
        void jvp_apic(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const ExtrapolationCache& extrapolation, const VectorField& position_tangent, const StaggeredField& extrapolated_velocity_tangent, VectorField& velocity_tangent, MatrixField& affine_tangent) const;
        void vjp_apic(cuda::Resource& resource, const Configuration& configuration, const VectorField& positions, const TransferCache& transfer, const ExtrapolationCache& extrapolation, const VectorAdjointField& velocity_adjoint, const MatrixAdjointField& affine_adjoint, VectorAdjointField& position_adjoint, StaggeredAdjointField& extrapolated_velocity_adjoint) const;
    };

    struct ParticleAdvectionOperator {
        void forward(cuda::Resource& resource, const Configuration& configuration, const DeviceDomain& domain, float particle_radius, const VectorField& positions, const VectorField& velocities, AdvectionCache& cache, VectorField& next_positions, VectorField& next_velocities) const;
        void jvp(cuda::Resource& resource, const Configuration& configuration, float particle_radius, const AdvectionCache& cache, const VectorField& position_tangent, const VectorField& velocity_tangent, VectorField& next_position_tangent, VectorField& next_velocity_tangent) const;
        void vjp(cuda::Resource& resource, const Configuration& configuration, float particle_radius, const AdvectionCache& cache, const VectorAdjointField& next_position_adjoint, const VectorAdjointField& next_velocity_adjoint, VectorAdjointField& position_adjoint, VectorAdjointField& velocity_adjoint) const;
    };

} // namespace xayah::fluid::grid
