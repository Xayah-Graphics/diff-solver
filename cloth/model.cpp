module;

#include "kernels.cuh"

module xayah.cloth.model;

import std;
import xayah.cloth.data;
import xayah.cloth.operators;
import xayah.cuda;

namespace xayah::cloth {

    namespace {

        float distance(const Vector3 first, const Vector3 second) {
            const float x = second.x - first.x;
            const float y = second.y - first.y;
            const float z = second.z - first.z;
            return std::sqrt(x * x + y * y + z * z);
        }

        void build_adjacency(const std::vector<Spring>& springs, const std::size_t particle_count, std::vector<std::uint32_t>& offsets, std::vector<std::uint32_t>& indices, std::vector<std::uint32_t>& others, std::vector<float>& signs) {
            offsets.assign(particle_count + 1, 0);
            for (const Spring& spring : springs) {
                ++offsets[spring.first + 1];
                ++offsets[spring.second + 1];
            }
            for (std::size_t particle = 1; particle < offsets.size(); ++particle) offsets[particle] += offsets[particle - 1];

            indices.resize(springs.size() * 2);
            others.resize(springs.size() * 2);
            signs.resize(springs.size() * 2);
            std::vector<std::uint32_t> cursors = offsets;
            for (std::uint32_t spring_index = 0; spring_index < springs.size(); ++spring_index) {
                const Spring& spring             = springs[spring_index];
                const std::uint32_t first_entry  = cursors[spring.first]++;
                indices[first_entry]             = spring_index;
                others[first_entry]              = spring.second;
                signs[first_entry]               = 1.0F;
                const std::uint32_t second_entry = cursors[spring.second]++;
                indices[second_entry]            = spring_index;
                others[second_entry]             = spring.first;
                signs[second_entry]              = -1.0F;
            }
        }

        Topology build_topology(const Configuration& configuration) {
            std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::uint32_t>> edge_opposites;
            for (const Triangle triangle : configuration.triangles) {
                const std::array vertices{triangle.first, triangle.second, triangle.third};
                for (std::size_t edge = 0; edge < 3; ++edge) {
                    const std::uint32_t first  = std::min(vertices[edge], vertices[(edge + 1) % 3]);
                    const std::uint32_t second = std::max(vertices[edge], vertices[(edge + 1) % 3]);
                    edge_opposites[{first, second}].push_back(vertices[(edge + 2) % 3]);
                }
            }

            Topology topology{};
            topology.stretch_springs.reserve(edge_opposites.size());
            for (const auto& [edge, opposites] : edge_opposites) {
                topology.stretch_springs.push_back({.first = edge.first, .second = edge.second, .rest_length = distance(configuration.rest_positions[edge.first], configuration.rest_positions[edge.second])});
                if (opposites.size() == 2) {
                    const std::uint32_t first  = std::min(opposites[0], opposites[1]);
                    const std::uint32_t second = std::max(opposites[0], opposites[1]);
                    topology.bending_springs.push_back({.first = first, .second = second, .rest_length = distance(configuration.rest_positions[first], configuration.rest_positions[second])});
                }
            }

            build_adjacency(topology.stretch_springs, configuration.rest_positions.size(), topology.stretch_offsets, topology.stretch_indices, topology.stretch_others, topology.stretch_signs);
            build_adjacency(topology.bending_springs, configuration.rest_positions.size(), topology.bending_offsets, topology.bending_indices, topology.bending_others, topology.bending_signs);
            return topology;
        }

    } // namespace

    ExecutionContext::ExecutionContext(const Configuration& configuration, const Topology& topology) : resource_owner_(std::make_shared<cuda::Resource>()), resource(*resource_owner_), device_topology{}, integrated_state_{}, force_tangent_{}, integrated_state_tangent_{}, force_adjoint_{}, integrated_state_adjoint_{} {
        device_topology.stretch = {
            .first   = make_index_buffer(topology.stretch_springs.size()),
            .second  = make_index_buffer(topology.stretch_springs.size()),
            .offsets = make_index_buffer(topology.stretch_offsets.size()),
            .indices = make_index_buffer(topology.stretch_indices.size()),
            .others  = make_index_buffer(topology.stretch_others.size()),
            .signs   = make_scalar_buffer(topology.stretch_signs.size()),
        };
        device_topology.bending = {
            .first   = make_index_buffer(topology.bending_springs.size()),
            .second  = make_index_buffer(topology.bending_springs.size()),
            .offsets = make_index_buffer(topology.bending_offsets.size()),
            .indices = make_index_buffer(topology.bending_indices.size()),
            .others  = make_index_buffer(topology.bending_others.size()),
            .signs   = make_scalar_buffer(topology.bending_signs.size()),
        };
        device_topology.anchor_mask      = make_index_buffer(configuration.anchors.size());
        device_topology.anchor_positions = make_vector_field(configuration.anchors.size());

        std::vector<std::uint32_t> stretch_first(topology.stretch_springs.size());
        std::vector<std::uint32_t> stretch_second(topology.stretch_springs.size());
        for (std::size_t spring = 0; spring < topology.stretch_springs.size(); ++spring) {
            stretch_first[spring]  = topology.stretch_springs[spring].first;
            stretch_second[spring] = topology.stretch_springs[spring].second;
        }
        std::vector<std::uint32_t> bending_first(topology.bending_springs.size());
        std::vector<std::uint32_t> bending_second(topology.bending_springs.size());
        for (std::size_t spring = 0; spring < topology.bending_springs.size(); ++spring) {
            bending_first[spring]  = topology.bending_springs[spring].first;
            bending_second[spring] = topology.bending_springs[spring].second;
        }

        if (!stretch_first.empty()) resource.copy_from_host(device_topology.stretch.first.data, stretch_first.data(), stretch_first.size() * sizeof(std::uint32_t));
        if (!stretch_second.empty()) resource.copy_from_host(device_topology.stretch.second.data, stretch_second.data(), stretch_second.size() * sizeof(std::uint32_t));
        resource.copy_from_host(device_topology.stretch.offsets.data, topology.stretch_offsets.data(), topology.stretch_offsets.size() * sizeof(std::uint32_t));
        if (!topology.stretch_indices.empty()) resource.copy_from_host(device_topology.stretch.indices.data, topology.stretch_indices.data(), topology.stretch_indices.size() * sizeof(std::uint32_t));
        if (!topology.stretch_others.empty()) resource.copy_from_host(device_topology.stretch.others.data, topology.stretch_others.data(), topology.stretch_others.size() * sizeof(std::uint32_t));
        if (!topology.stretch_signs.empty()) resource.copy_from_host(device_topology.stretch.signs.data, topology.stretch_signs.data(), topology.stretch_signs.size() * sizeof(float));
        if (!bending_first.empty()) resource.copy_from_host(device_topology.bending.first.data, bending_first.data(), bending_first.size() * sizeof(std::uint32_t));
        if (!bending_second.empty()) resource.copy_from_host(device_topology.bending.second.data, bending_second.data(), bending_second.size() * sizeof(std::uint32_t));
        resource.copy_from_host(device_topology.bending.offsets.data, topology.bending_offsets.data(), topology.bending_offsets.size() * sizeof(std::uint32_t));
        if (!topology.bending_indices.empty()) resource.copy_from_host(device_topology.bending.indices.data, topology.bending_indices.data(), topology.bending_indices.size() * sizeof(std::uint32_t));
        if (!topology.bending_others.empty()) resource.copy_from_host(device_topology.bending.others.data, topology.bending_others.data(), topology.bending_others.size() * sizeof(std::uint32_t));
        if (!topology.bending_signs.empty()) resource.copy_from_host(device_topology.bending.signs.data, topology.bending_signs.data(), topology.bending_signs.size() * sizeof(float));

        std::vector<std::uint32_t> anchor_mask(configuration.anchors.size());
        std::vector<Vector3> anchor_positions(configuration.anchors.size());
        for (std::size_t particle = 0; particle < configuration.anchors.size(); ++particle) {
            anchor_mask[particle]      = configuration.anchors[particle].has_value() ? 1U : 0U;
            anchor_positions[particle] = configuration.anchors[particle].value_or(configuration.rest_positions[particle]);
        }
        resource.copy_from_host(device_topology.anchor_mask.data, anchor_mask.data(), anchor_mask.size() * sizeof(std::uint32_t));
        std::vector<float> x(anchor_positions.size());
        std::vector<float> y(anchor_positions.size());
        std::vector<float> z(anchor_positions.size());
        for (std::size_t particle = 0; particle < anchor_positions.size(); ++particle) {
            x[particle] = anchor_positions[particle].x;
            y[particle] = anchor_positions[particle].y;
            z[particle] = anchor_positions[particle].z;
        }
        resource.copy_from_host(device_topology.anchor_positions.x.data, x.data(), x.size() * sizeof(float));
        resource.copy_from_host(device_topology.anchor_positions.y.data, y.data(), y.size() * sizeof(float));
        resource.copy_from_host(device_topology.anchor_positions.z.data, z.data(), z.size() * sizeof(float));

        integrated_state_         = {.positions = make_vector_field(configuration.rest_positions.size()), .velocities = make_vector_field(configuration.rest_positions.size())};
        force_tangent_            = {.values = make_vector_field(configuration.rest_positions.size())};
        integrated_state_tangent_ = {.positions = make_vector_field(configuration.rest_positions.size()), .velocities = make_vector_field(configuration.rest_positions.size())};
        force_adjoint_            = {.values = make_vector_field(configuration.rest_positions.size())};
        integrated_state_adjoint_ = {.positions = make_vector_field(configuration.rest_positions.size()), .velocities = make_vector_field(configuration.rest_positions.size())};
        resource.synchronize();
    }

    void ExecutionContext::upload(const std::span<const float> source, cuda::Buffer<float>& destination) {
        if (!source.empty()) resource.copy_from_host(destination.data, source.data(), source.size_bytes());
        resource.synchronize();
    }

    void ExecutionContext::download(const cuda::Buffer<float>& source, const std::span<float> destination) {
        if (!destination.empty()) resource.copy_to_host(destination.data(), source.data, destination.size_bytes());
        resource.synchronize();
    }

    void ExecutionContext::upload(const std::span<const Vector3> source, VectorField& destination) {
        std::vector<float> x(source.size());
        std::vector<float> y(source.size());
        std::vector<float> z(source.size());
        for (std::size_t index = 0; index < source.size(); ++index) {
            x[index] = source[index].x;
            y[index] = source[index].y;
            z[index] = source[index].z;
        }
        if (!source.empty()) {
            resource.copy_from_host(destination.x.data, x.data(), x.size() * sizeof(float));
            resource.copy_from_host(destination.y.data, y.data(), y.size() * sizeof(float));
            resource.copy_from_host(destination.z.data, z.data(), z.size() * sizeof(float));
        }
        resource.synchronize();
    }

    void ExecutionContext::download(const VectorField& source, const std::span<Vector3> destination) {
        std::vector<float> x(destination.size());
        std::vector<float> y(destination.size());
        std::vector<float> z(destination.size());
        if (!destination.empty()) {
            resource.copy_to_host(x.data(), source.x.data, x.size() * sizeof(float));
            resource.copy_to_host(y.data(), source.y.data, y.size() * sizeof(float));
            resource.copy_to_host(z.data(), source.z.data, z.size() * sizeof(float));
        }
        resource.synchronize();
        for (std::size_t index = 0; index < destination.size(); ++index) destination[index] = {.x = x[index], .y = y[index], .z = z[index]};
    }

    void ExecutionContext::synchronize() {
        resource.synchronize();
    }

    cuda::Buffer<float> ExecutionContext::make_scalar_buffer(const std::size_t size) const {
        return cuda::Buffer<float>(resource_owner_, size);
    }

    cuda::Buffer<std::uint32_t> ExecutionContext::make_index_buffer(const std::size_t size) const {
        return cuda::Buffer<std::uint32_t>(resource_owner_, size);
    }

    VectorField ExecutionContext::make_vector_field(const std::size_t size) const {
        return {.x = make_scalar_buffer(size), .y = make_scalar_buffer(size), .z = make_scalar_buffer(size)};
    }

    void ExecutionContext::zero(cuda::Buffer<float>& buffer) {
        if (buffer.size != 0) resource.zero(buffer.data, buffer.size * sizeof(float));
    }

    void ExecutionContext::zero(VectorField& field) {
        zero(field.x);
        zero(field.y);
        zero(field.z);
    }

    void ExecutionContext::copy(const cuda::Buffer<float>& source, cuda::Buffer<float>& destination) {
        if (source.size != 0) resource.copy_device(destination.data, source.data, source.size * sizeof(float));
    }

    void ExecutionContext::copy(const VectorField& source, VectorField& destination) {
        copy(source.x, destination.x);
        copy(source.y, destination.y);
        copy(source.z, destination.z);
    }

    void ExecutionContext::accumulate(const VectorField& source, VectorField& destination) {
        cuda_kernel::launch_accumulate(resource.native_stream, static_cast<std::uint32_t>(source.x.size), {.x = source.x.data, .y = source.y.data, .z = source.z.data}, {.x = destination.x.data, .y = destination.y.data, .z = destination.z.data});
    }

    Model::Model(Configuration next_configuration) : configuration(std::move(next_configuration)), topology(build_topology(configuration)) {}

    ExecutionContext Model::make_context() const {
        return ExecutionContext(configuration, topology);
    }

    State Model::make_state(ExecutionContext& context) const {
        return {.positions = context.make_vector_field(configuration.rest_positions.size()), .velocities = context.make_vector_field(configuration.rest_positions.size())};
    }

    Control Model::make_control(ExecutionContext& context) const {
        Control control{.external_forces = context.make_vector_field(configuration.rest_positions.size())};
        context.zero(control.external_forces);
        return control;
    }

    Parameters Model::make_parameters(ExecutionContext& context) const {
        Parameters parameters{
            .masses               = context.make_scalar_buffer(configuration.rest_positions.size()),
            .stretch_stiffnesses  = context.make_scalar_buffer(topology.stretch_springs.size()),
            .stretch_dampings     = context.make_scalar_buffer(topology.stretch_springs.size()),
            .stretch_rest_lengths = context.make_scalar_buffer(topology.stretch_springs.size()),
            .bending_stiffnesses  = context.make_scalar_buffer(topology.bending_springs.size()),
            .bending_dampings     = context.make_scalar_buffer(topology.bending_springs.size()),
            .bending_rest_lengths = context.make_scalar_buffer(topology.bending_springs.size()),
        };
        context.zero(parameters.masses);
        context.zero(parameters.stretch_stiffnesses);
        context.zero(parameters.stretch_dampings);
        context.zero(parameters.stretch_rest_lengths);
        context.zero(parameters.bending_stiffnesses);
        context.zero(parameters.bending_dampings);
        context.zero(parameters.bending_rest_lengths);
        return parameters;
    }

    StepCache Model::make_step_cache(ExecutionContext& context) const {
        return {.forces = {.values = context.make_vector_field(configuration.rest_positions.size())}};
    }

    StateTangent Model::make_state_tangent(ExecutionContext& context) const {
        StateTangent tangent{.positions = context.make_vector_field(configuration.rest_positions.size()), .velocities = context.make_vector_field(configuration.rest_positions.size())};
        context.zero(tangent.positions);
        context.zero(tangent.velocities);
        return tangent;
    }

    ControlTangent Model::make_control_tangent(ExecutionContext& context) const {
        ControlTangent tangent{.external_forces = context.make_vector_field(configuration.rest_positions.size())};
        context.zero(tangent.external_forces);
        return tangent;
    }

    ParameterTangent Model::make_parameter_tangent(ExecutionContext& context) const {
        ParameterTangent tangent{
            .masses               = context.make_scalar_buffer(configuration.rest_positions.size()),
            .stretch_stiffnesses  = context.make_scalar_buffer(topology.stretch_springs.size()),
            .stretch_dampings     = context.make_scalar_buffer(topology.stretch_springs.size()),
            .stretch_rest_lengths = context.make_scalar_buffer(topology.stretch_springs.size()),
            .bending_stiffnesses  = context.make_scalar_buffer(topology.bending_springs.size()),
            .bending_dampings     = context.make_scalar_buffer(topology.bending_springs.size()),
            .bending_rest_lengths = context.make_scalar_buffer(topology.bending_springs.size()),
        };
        context.zero(tangent.masses);
        context.zero(tangent.stretch_stiffnesses);
        context.zero(tangent.stretch_dampings);
        context.zero(tangent.stretch_rest_lengths);
        context.zero(tangent.bending_stiffnesses);
        context.zero(tangent.bending_dampings);
        context.zero(tangent.bending_rest_lengths);
        return tangent;
    }

    StateAdjoint Model::make_state_adjoint(ExecutionContext& context) const {
        StateAdjoint adjoint{.positions = context.make_vector_field(configuration.rest_positions.size()), .velocities = context.make_vector_field(configuration.rest_positions.size())};
        context.zero(adjoint.positions);
        context.zero(adjoint.velocities);
        return adjoint;
    }

    ControlAdjoint Model::make_control_adjoint(ExecutionContext& context) const {
        ControlAdjoint adjoint{.external_forces = context.make_vector_field(configuration.rest_positions.size())};
        context.zero(adjoint.external_forces);
        return adjoint;
    }

    ParameterAdjoint Model::make_parameter_adjoint(ExecutionContext& context) const {
        ParameterAdjoint adjoint{
            .masses               = context.make_scalar_buffer(configuration.rest_positions.size()),
            .stretch_stiffnesses  = context.make_scalar_buffer(topology.stretch_springs.size()),
            .stretch_dampings     = context.make_scalar_buffer(topology.stretch_springs.size()),
            .stretch_rest_lengths = context.make_scalar_buffer(topology.stretch_springs.size()),
            .bending_stiffnesses  = context.make_scalar_buffer(topology.bending_springs.size()),
            .bending_dampings     = context.make_scalar_buffer(topology.bending_springs.size()),
            .bending_rest_lengths = context.make_scalar_buffer(topology.bending_springs.size()),
        };
        context.zero(adjoint.masses);
        context.zero(adjoint.stretch_stiffnesses);
        context.zero(adjoint.stretch_dampings);
        context.zero(adjoint.stretch_rest_lengths);
        context.zero(adjoint.bending_stiffnesses);
        context.zero(adjoint.bending_dampings);
        context.zero(adjoint.bending_rest_lengths);
        return adjoint;
    }

    void Model::copy_state(const State& source, State& destination, ExecutionContext& context) const {
        context.copy(source.positions, destination.positions);
        context.copy(source.velocities, destination.velocities);
    }

    void Model::copy_state_tangent(const StateTangent& source, StateTangent& destination, ExecutionContext& context) const {
        context.copy(source.positions, destination.positions);
        context.copy(source.velocities, destination.velocities);
    }

    void Model::copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        context.copy(source.positions, destination.positions);
        context.copy(source.velocities, destination.velocities);
    }

    void Model::accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        context.accumulate(source.positions, destination.positions);
        context.accumulate(source.velocities, destination.velocities);
    }

    void Model::forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& step_cache, ExecutionContext& context) const {
        force_assembly_.forward(context.resource, context.device_topology, configuration, state, control, parameters, step_cache.forces);
        semi_implicit_euler_.forward(context.resource, configuration, state, parameters, step_cache.forces, context.integrated_state_);
        fixed_constraint_.forward(context.resource, context.device_topology, context.integrated_state_, next_state);
    }

    void Model::jvp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const {
        force_assembly_.jvp(context.resource, context.device_topology, configuration, state, parameters, state_tangent, control_tangent, parameter_tangent, context.force_tangent_);
        semi_implicit_euler_.jvp(context.resource, configuration, parameters, step_cache.forces, state_tangent, parameter_tangent, context.force_tangent_, context.integrated_state_tangent_);
        fixed_constraint_.jvp(context.resource, context.device_topology, context.integrated_state_tangent_, next_state_tangent);
    }

    void Model::vjp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const {
        context.zero(context.force_adjoint_.values);
        context.zero(context.integrated_state_adjoint_.positions);
        context.zero(context.integrated_state_adjoint_.velocities);
        fixed_constraint_.vjp(context.resource, context.device_topology, next_state_adjoint, context.integrated_state_adjoint_);
        semi_implicit_euler_.vjp(context.resource, configuration, parameters, step_cache.forces, context.integrated_state_adjoint_, previous_state_adjoint, context.force_adjoint_, parameter_adjoint);
        force_assembly_.vjp(context.resource, context.device_topology, configuration, state, parameters, context.force_adjoint_, previous_state_adjoint, control_adjoint, parameter_adjoint);
    }

} // namespace xayah::cloth
