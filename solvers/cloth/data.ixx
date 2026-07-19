export module xayah.cloth.data;

import std;
import xayah.cuda;

export namespace xayah::cloth {

    struct Vector3 {
        float x;
        float y;
        float z;
    };

    struct Triangle {
        std::uint32_t first;
        std::uint32_t second;
        std::uint32_t third;
    };

    struct Spring {
        std::uint32_t first;
        std::uint32_t second;
        float rest_length;
    };

    struct Configuration {
        std::vector<Vector3> rest_positions;
        std::vector<Triangle> triangles;
        std::vector<std::optional<Vector3>> anchors;
        Vector3 gravity;
        float time_step;
    };

    struct Topology {
        std::vector<Spring> stretch_springs;
        std::vector<Spring> bending_springs;
        std::vector<std::uint32_t> stretch_offsets;
        std::vector<std::uint32_t> stretch_indices;
        std::vector<std::uint32_t> stretch_others;
        std::vector<float> stretch_signs;
        std::vector<std::uint32_t> bending_offsets;
        std::vector<std::uint32_t> bending_indices;
        std::vector<std::uint32_t> bending_others;
        std::vector<float> bending_signs;
    };

    struct VectorField {
        cuda::Buffer<float> x;
        cuda::Buffer<float> y;
        cuda::Buffer<float> z;
    };

    struct State {
        VectorField positions;
        VectorField velocities;
    };

    struct Control {
        VectorField external_forces;
    };

    struct Parameters {
        cuda::Buffer<float> masses;
        cuda::Buffer<float> stretch_stiffnesses;
        cuda::Buffer<float> stretch_dampings;
        cuda::Buffer<float> stretch_rest_lengths;
        cuda::Buffer<float> bending_stiffnesses;
        cuda::Buffer<float> bending_dampings;
        cuda::Buffer<float> bending_rest_lengths;
    };

    struct Forces {
        VectorField values;
    };

    struct StateTangent {
        VectorField positions;
        VectorField velocities;
    };

    struct ControlTangent {
        VectorField external_forces;
    };

    struct ParameterTangent {
        cuda::Buffer<float> masses;
        cuda::Buffer<float> stretch_stiffnesses;
        cuda::Buffer<float> stretch_dampings;
        cuda::Buffer<float> stretch_rest_lengths;
        cuda::Buffer<float> bending_stiffnesses;
        cuda::Buffer<float> bending_dampings;
        cuda::Buffer<float> bending_rest_lengths;
    };

    struct ForceTangent {
        VectorField values;
    };

    struct StateAdjoint {
        VectorField positions;
        VectorField velocities;
    };

    struct ControlAdjoint {
        VectorField external_forces;
    };

    struct ParameterAdjoint {
        cuda::Buffer<float> masses;
        cuda::Buffer<float> stretch_stiffnesses;
        cuda::Buffer<float> stretch_dampings;
        cuda::Buffer<float> stretch_rest_lengths;
        cuda::Buffer<float> bending_stiffnesses;
        cuda::Buffer<float> bending_dampings;
        cuda::Buffer<float> bending_rest_lengths;
    };

    struct ForceAdjoint {
        VectorField values;
    };

    struct StepCache {
        Forces forces;
    };

    struct DeviceSpringTopology {
        cuda::Buffer<std::uint32_t> first;
        cuda::Buffer<std::uint32_t> second;
        cuda::Buffer<std::uint32_t> offsets;
        cuda::Buffer<std::uint32_t> indices;
        cuda::Buffer<std::uint32_t> others;
        cuda::Buffer<float> signs;
    };

    struct DeviceTopology {
        DeviceSpringTopology stretch;
        DeviceSpringTopology bending;
        cuda::Buffer<std::uint32_t> anchor_mask;
        VectorField anchor_positions;
    };

} // namespace xayah::cloth
