export module xayah.fluid.data;

import std;
import xayah.cuda;

export namespace xayah::fluid {

    struct Vector3 {
        float x;
        float y;
        float z;
    };

    struct BoxBoundary {
        Vector3 minimum;
        Vector3 maximum;
        Vector3 velocity;
        bool no_slip{true};
    };

    struct BoundaryParticle {
        Vector3 position;
        Vector3 velocity;
        float volume;
    };

    struct Configuration {
        std::uint32_t particle_count;
        float time_step;
        float support_radius;
        float particle_radius;
        Vector3 gravity;
        BoxBoundary domain;
        std::vector<BoundaryParticle> boundary_particles;
    };

    enum class ExecutionMode : std::uint32_t {
        forward,
        differentiable,
    };

    struct ScalarField {
        cuda::Buffer<float> values;
    };

    struct ScalarAdjointField {
        cuda::Buffer<double> values;
    };

    struct VectorField {
        cuda::Buffer<float> x;
        cuda::Buffer<float> y;
        cuda::Buffer<float> z;
    };

    struct VectorAdjointField {
        cuda::Buffer<double> x;
        cuda::Buffer<double> y;
        cuda::Buffer<double> z;
    };

    struct MatrixField {
        cuda::Buffer<float> xx;
        cuda::Buffer<float> xy;
        cuda::Buffer<float> xz;
        cuda::Buffer<float> yx;
        cuda::Buffer<float> yy;
        cuda::Buffer<float> yz;
        cuda::Buffer<float> zx;
        cuda::Buffer<float> zy;
        cuda::Buffer<float> zz;
    };

    struct MatrixAdjointField {
        cuda::Buffer<double> xx;
        cuda::Buffer<double> xy;
        cuda::Buffer<double> xz;
        cuda::Buffer<double> yx;
        cuda::Buffer<double> yy;
        cuda::Buffer<double> yz;
        cuda::Buffer<double> zx;
        cuda::Buffer<double> zy;
        cuda::Buffer<double> zz;
    };

    struct State {
        VectorField positions;
        VectorField velocities;
        std::uint64_t step_index{};
    };

    struct Control {
        VectorField external_accelerations;
    };

    struct StateTangent {
        VectorField positions;
        VectorField velocities;
    };

    struct ControlTangent {
        VectorField external_accelerations;
    };

    struct StateAdjoint {
        VectorAdjointField positions;
        VectorAdjointField velocities;
    };

    struct ControlAdjoint {
        VectorAdjointField external_accelerations;
    };

    struct ParticleParameters {
        cuda::Buffer<float> masses;
        cuda::Buffer<float> rest_densities;
        cuda::Buffer<float> viscosities;
        cuda::Buffer<float> surface_tensions;
    };

    struct ParticleParameterTangent {
        cuda::Buffer<float> masses;
        cuda::Buffer<float> rest_densities;
        cuda::Buffer<float> viscosities;
        cuda::Buffer<float> surface_tensions;
    };

    struct ParticleParameterAdjoint {
        cuda::Buffer<double> masses;
        cuda::Buffer<double> rest_densities;
        cuda::Buffer<double> viscosities;
        cuda::Buffer<double> surface_tensions;
    };

    struct Neighborhood {
        cuda::Buffer<std::uint64_t> sorted_keys;
        cuda::Buffer<std::uint32_t> sorted_particle_indices;
        cuda::Buffer<std::uint32_t> cell_offsets;
        cuda::Buffer<std::uint64_t> sorted_boundary_keys;
        cuda::Buffer<std::uint32_t> sorted_boundary_indices;
        cuda::Buffer<std::uint32_t> boundary_cell_offsets;
        std::array<std::uint32_t, 3u> cell_resolution;
        Vector3 cell_origin;
        float cell_size;
        float boundary_time;
    };

    struct NeighborSearch {
        cuda::Buffer<std::uint64_t> unsorted_keys;
        cuda::Buffer<std::uint32_t> unsorted_particle_indices;
        cuda::Buffer<std::uint64_t> unsorted_boundary_keys;
        cuda::Buffer<std::uint32_t> unsorted_boundary_indices;
        cuda::Buffer<std::byte> sort_scratch;
        cuda::Buffer<std::byte> boundary_sort_scratch;
    };

} // namespace xayah::fluid
