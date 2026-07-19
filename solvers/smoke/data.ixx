export module xayah.smoke.data;

import std;
import xayah.cuda;

export namespace xayah::smoke {

    struct Vector3 {
        float x;
        float y;
        float z;
    };

    enum class ScalarBoundaryMode : std::uint32_t {
        fixed_value,
        zero_gradient,
        periodic,
    };

    enum class VelocityBoundaryMode : std::uint32_t {
        fixed_value,
        zero_gradient,
        normal_fixed_tangent_zero_gradient,
        periodic,
    };

    struct ScalarBoundaryFace {
        ScalarBoundaryMode mode{ScalarBoundaryMode::zero_gradient};
        float value{};
    };

    struct VelocityBoundaryFace {
        VelocityBoundaryMode mode{VelocityBoundaryMode::zero_gradient};
        Vector3 value{};
    };

    struct ScalarBoundary {
        ScalarBoundaryFace x_min{};
        ScalarBoundaryFace x_max{};
        ScalarBoundaryFace y_min{};
        ScalarBoundaryFace y_max{};
        ScalarBoundaryFace z_min{};
        ScalarBoundaryFace z_max{};
    };

    struct VelocityBoundary {
        VelocityBoundaryFace x_min{};
        VelocityBoundaryFace x_max{};
        VelocityBoundaryFace y_min{};
        VelocityBoundaryFace y_max{};
        VelocityBoundaryFace z_min{};
        VelocityBoundaryFace z_max{};
    };

    struct Ellipsoid {
        Vector3 center{};
        Vector3 radius{1.0F, 1.0F, 1.0F};
    };

    struct Box {
        Vector3 center{};
        Vector3 half_extent{1.0F, 1.0F, 1.0F};
    };

    struct Collider {
        std::variant<Ellipsoid, Box> shape{Ellipsoid{}};
        Vector3 velocity{};
        float density{};
        float temperature{};
    };

    struct Configuration {
        std::array<std::uint32_t, 3u> resolution{32u, 48u, 32u};
        float cell_size{1.0F / 32.0F};
        float time_step{1.0F / 60.0F};
        std::uint32_t pressure_iterations{80u};
        VelocityBoundary velocity_boundary{};
        ScalarBoundary pressure_boundary{};
        ScalarBoundary density_boundary{};
        ScalarBoundary temperature_boundary{};
        std::vector<Collider> colliders{};
        bool vorticity_confinement_enabled{true};
    };

    struct ScalarField {
        cuda::Buffer<float> values;
    };

    struct CenteredVectorField {
        ScalarField x;
        ScalarField y;
        ScalarField z;
    };

    struct StaggeredVectorField {
        cuda::Buffer<float> x;
        cuda::Buffer<float> y;
        cuda::Buffer<float> z;
    };

    struct ScalarAdjointField {
        cuda::Buffer<double> values;
    };

    struct CenteredVectorAdjointField {
        ScalarAdjointField x;
        ScalarAdjointField y;
        ScalarAdjointField z;
    };

    struct StaggeredVectorAdjointField {
        cuda::Buffer<double> x;
        cuda::Buffer<double> y;
        cuda::Buffer<double> z;
    };

    struct State {
        ScalarField density;
        ScalarField temperature;
        StaggeredVectorField velocity;
    };

    struct Control {
        ScalarField density_source;
        ScalarField temperature_source;
        CenteredVectorField external_acceleration;
    };

    struct Parameters {
        cuda::Buffer<float> ambient_temperature;
        cuda::Buffer<float> density_buoyancy;
        cuda::Buffer<float> temperature_buoyancy;
        cuda::Buffer<float> vorticity_confinement;
    };

    struct StateTangent {
        ScalarField density;
        ScalarField temperature;
        StaggeredVectorField velocity;
    };

    struct ControlTangent {
        ScalarField density_source;
        ScalarField temperature_source;
        CenteredVectorField external_acceleration;
    };

    struct ParameterTangent {
        cuda::Buffer<float> ambient_temperature;
        cuda::Buffer<float> density_buoyancy;
        cuda::Buffer<float> temperature_buoyancy;
        cuda::Buffer<float> vorticity_confinement;
    };

    struct StateAdjoint {
        ScalarAdjointField density;
        ScalarAdjointField temperature;
        StaggeredVectorAdjointField velocity;
    };

    struct ControlAdjoint {
        ScalarAdjointField density_source;
        ScalarAdjointField temperature_source;
        CenteredVectorAdjointField external_acceleration;
    };

    struct ParameterAdjoint {
        cuda::Buffer<double> ambient_temperature;
        cuda::Buffer<double> density_buoyancy;
        cuda::Buffer<double> temperature_buoyancy;
        cuda::Buffer<double> vorticity_confinement;
    };

    struct VorticityCache {
        CenteredVectorField centered_velocity;
        CenteredVectorField vorticity;
        ScalarField magnitude;
        CenteredVectorField normal;
        ScalarField normalizer;
    };

    struct VorticityAdjointCache {
        CenteredVectorAdjointField centered_velocity;
        CenteredVectorAdjointField vorticity;
        ScalarAdjointField magnitude;
        CenteredVectorAdjointField normal;
    };

    struct StepCache {
        ScalarField sourced_density;
        ScalarField sourced_temperature;
        CenteredVectorField force;
        VorticityCache vorticity;
        StaggeredVectorField forced_velocity;
        StaggeredVectorField advected_velocity;
    };

    struct DeviceDomain {
        cuda::Buffer<std::uint32_t> cell_mask;
        StaggeredVectorField collider_velocity;
        ScalarField collider_density;
        ScalarField collider_temperature;
        std::uint32_t pressure_anchor{};
    };

} // namespace xayah::smoke
