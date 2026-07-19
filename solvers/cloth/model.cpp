module;

#include "operators.h"

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

    void upload(cuda::Resource& resource, const std::span<const float> source, cuda::Buffer<float>& destination) {
        if (!source.empty()) resource.copy_from_host(destination.data, source.data(), source.size_bytes());
        resource.synchronize();
    }

    void upload(cuda::Resource& resource, const std::span<const Vector3> source, VectorField& destination) {
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

    void download(cuda::Resource& resource, const VectorField& source, const std::span<Vector3> destination) {
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

    Model::Model(Configuration next_configuration) : configuration(std::move(next_configuration)), topology(build_topology(configuration)) {}

    ExecutionContext Model::allocate_context(const ExecutionMode mode) const {
        ExecutionContext context{};
        context.resource                = std::make_shared<cuda::Resource>();
        cuda::Resource& resource        = *context.resource;
        DeviceTopology& device_topology = context.device_topology;
        device_topology.stretch         = {
                    .first   = cuda::Buffer<std::uint32_t>(context.resource, topology.stretch_springs.size()),
                    .second  = cuda::Buffer<std::uint32_t>(context.resource, topology.stretch_springs.size()),
                    .offsets = cuda::Buffer<std::uint32_t>(context.resource, topology.stretch_offsets.size()),
                    .indices = cuda::Buffer<std::uint32_t>(context.resource, topology.stretch_indices.size()),
                    .others  = cuda::Buffer<std::uint32_t>(context.resource, topology.stretch_others.size()),
                    .signs   = cuda::Buffer<float>(context.resource, topology.stretch_signs.size()),
        };
        device_topology.bending = {
            .first   = cuda::Buffer<std::uint32_t>(context.resource, topology.bending_springs.size()),
            .second  = cuda::Buffer<std::uint32_t>(context.resource, topology.bending_springs.size()),
            .offsets = cuda::Buffer<std::uint32_t>(context.resource, topology.bending_offsets.size()),
            .indices = cuda::Buffer<std::uint32_t>(context.resource, topology.bending_indices.size()),
            .others  = cuda::Buffer<std::uint32_t>(context.resource, topology.bending_others.size()),
            .signs   = cuda::Buffer<float>(context.resource, topology.bending_signs.size()),
        };
        device_topology.anchor_mask      = cuda::Buffer<std::uint32_t>(context.resource, configuration.anchors.size());
        device_topology.anchor_positions = allocate_vector_field(context, configuration.anchors.size());

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

        context.integrated_state_ = {.positions = allocate_vector_field(context, configuration.rest_positions.size()), .velocities = allocate_vector_field(context, configuration.rest_positions.size())};
        if (mode == ExecutionMode::differentiable) {
            context.force_tangent_            = {.values = allocate_vector_field(context, configuration.rest_positions.size())};
            context.integrated_state_tangent_ = {.positions = allocate_vector_field(context, configuration.rest_positions.size()), .velocities = allocate_vector_field(context, configuration.rest_positions.size())};
            context.force_adjoint_            = {.values = allocate_vector_field(context, configuration.rest_positions.size())};
            context.integrated_state_adjoint_ = {.positions = allocate_vector_field(context, configuration.rest_positions.size()), .velocities = allocate_vector_field(context, configuration.rest_positions.size())};
        }
        resource.synchronize();
        return context;
    }

    VectorField Model::allocate_vector_field(ExecutionContext& context, const std::size_t size) const {
        return {.x = cuda::Buffer<float>(context.resource, size), .y = cuda::Buffer<float>(context.resource, size), .z = cuda::Buffer<float>(context.resource, size)};
    }

    State Model::allocate_state(ExecutionContext& context) const {
        return {.positions = allocate_vector_field(context, configuration.rest_positions.size()), .velocities = allocate_vector_field(context, configuration.rest_positions.size())};
    }

    Control Model::allocate_control(ExecutionContext& context) const {
        Control control{.external_forces = allocate_vector_field(context, configuration.rest_positions.size())};
        context.resource->zero(control.external_forces.x.data, control.external_forces.x.size * sizeof(float));
        context.resource->zero(control.external_forces.y.data, control.external_forces.y.size * sizeof(float));
        context.resource->zero(control.external_forces.z.data, control.external_forces.z.size * sizeof(float));
        return control;
    }

    Parameters Model::allocate_parameters(ExecutionContext& context) const {
        Parameters parameters{
            .masses               = cuda::Buffer<float>(context.resource, configuration.rest_positions.size()),
            .stretch_stiffnesses  = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .stretch_dampings     = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .stretch_rest_lengths = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .bending_stiffnesses  = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
            .bending_dampings     = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
            .bending_rest_lengths = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
        };
        if (parameters.masses.size != 0u) context.resource->zero(parameters.masses.data, parameters.masses.size * sizeof(float));
        if (parameters.stretch_stiffnesses.size != 0u) context.resource->zero(parameters.stretch_stiffnesses.data, parameters.stretch_stiffnesses.size * sizeof(float));
        if (parameters.stretch_dampings.size != 0u) context.resource->zero(parameters.stretch_dampings.data, parameters.stretch_dampings.size * sizeof(float));
        if (parameters.stretch_rest_lengths.size != 0u) context.resource->zero(parameters.stretch_rest_lengths.data, parameters.stretch_rest_lengths.size * sizeof(float));
        if (parameters.bending_stiffnesses.size != 0u) context.resource->zero(parameters.bending_stiffnesses.data, parameters.bending_stiffnesses.size * sizeof(float));
        if (parameters.bending_dampings.size != 0u) context.resource->zero(parameters.bending_dampings.data, parameters.bending_dampings.size * sizeof(float));
        if (parameters.bending_rest_lengths.size != 0u) context.resource->zero(parameters.bending_rest_lengths.data, parameters.bending_rest_lengths.size * sizeof(float));
        return parameters;
    }

    StepCache Model::allocate_step_cache(ExecutionContext& context) const {
        return {.forces = {.values = allocate_vector_field(context, configuration.rest_positions.size())}};
    }

    StateTangent Model::allocate_state_tangent(ExecutionContext& context) const {
        StateTangent tangent{.positions = allocate_vector_field(context, configuration.rest_positions.size()), .velocities = allocate_vector_field(context, configuration.rest_positions.size())};
        context.resource->zero(tangent.positions.x.data, tangent.positions.x.size * sizeof(float));
        context.resource->zero(tangent.positions.y.data, tangent.positions.y.size * sizeof(float));
        context.resource->zero(tangent.positions.z.data, tangent.positions.z.size * sizeof(float));
        context.resource->zero(tangent.velocities.x.data, tangent.velocities.x.size * sizeof(float));
        context.resource->zero(tangent.velocities.y.data, tangent.velocities.y.size * sizeof(float));
        context.resource->zero(tangent.velocities.z.data, tangent.velocities.z.size * sizeof(float));
        return tangent;
    }

    ControlTangent Model::allocate_control_tangent(ExecutionContext& context) const {
        ControlTangent tangent{.external_forces = allocate_vector_field(context, configuration.rest_positions.size())};
        context.resource->zero(tangent.external_forces.x.data, tangent.external_forces.x.size * sizeof(float));
        context.resource->zero(tangent.external_forces.y.data, tangent.external_forces.y.size * sizeof(float));
        context.resource->zero(tangent.external_forces.z.data, tangent.external_forces.z.size * sizeof(float));
        return tangent;
    }

    ParameterTangent Model::allocate_parameter_tangent(ExecutionContext& context) const {
        ParameterTangent tangent{
            .masses               = cuda::Buffer<float>(context.resource, configuration.rest_positions.size()),
            .stretch_stiffnesses  = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .stretch_dampings     = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .stretch_rest_lengths = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .bending_stiffnesses  = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
            .bending_dampings     = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
            .bending_rest_lengths = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
        };
        if (tangent.masses.size != 0u) context.resource->zero(tangent.masses.data, tangent.masses.size * sizeof(float));
        if (tangent.stretch_stiffnesses.size != 0u) context.resource->zero(tangent.stretch_stiffnesses.data, tangent.stretch_stiffnesses.size * sizeof(float));
        if (tangent.stretch_dampings.size != 0u) context.resource->zero(tangent.stretch_dampings.data, tangent.stretch_dampings.size * sizeof(float));
        if (tangent.stretch_rest_lengths.size != 0u) context.resource->zero(tangent.stretch_rest_lengths.data, tangent.stretch_rest_lengths.size * sizeof(float));
        if (tangent.bending_stiffnesses.size != 0u) context.resource->zero(tangent.bending_stiffnesses.data, tangent.bending_stiffnesses.size * sizeof(float));
        if (tangent.bending_dampings.size != 0u) context.resource->zero(tangent.bending_dampings.data, tangent.bending_dampings.size * sizeof(float));
        if (tangent.bending_rest_lengths.size != 0u) context.resource->zero(tangent.bending_rest_lengths.data, tangent.bending_rest_lengths.size * sizeof(float));
        return tangent;
    }

    StateAdjoint Model::allocate_state_adjoint(ExecutionContext& context) const {
        StateAdjoint adjoint{.positions = allocate_vector_field(context, configuration.rest_positions.size()), .velocities = allocate_vector_field(context, configuration.rest_positions.size())};
        context.resource->zero(adjoint.positions.x.data, adjoint.positions.x.size * sizeof(float));
        context.resource->zero(adjoint.positions.y.data, adjoint.positions.y.size * sizeof(float));
        context.resource->zero(adjoint.positions.z.data, adjoint.positions.z.size * sizeof(float));
        context.resource->zero(adjoint.velocities.x.data, adjoint.velocities.x.size * sizeof(float));
        context.resource->zero(adjoint.velocities.y.data, adjoint.velocities.y.size * sizeof(float));
        context.resource->zero(adjoint.velocities.z.data, adjoint.velocities.z.size * sizeof(float));
        return adjoint;
    }

    ControlAdjoint Model::allocate_control_adjoint(ExecutionContext& context) const {
        ControlAdjoint adjoint{.external_forces = allocate_vector_field(context, configuration.rest_positions.size())};
        context.resource->zero(adjoint.external_forces.x.data, adjoint.external_forces.x.size * sizeof(float));
        context.resource->zero(adjoint.external_forces.y.data, adjoint.external_forces.y.size * sizeof(float));
        context.resource->zero(adjoint.external_forces.z.data, adjoint.external_forces.z.size * sizeof(float));
        return adjoint;
    }

    ParameterAdjoint Model::allocate_parameter_adjoint(ExecutionContext& context) const {
        ParameterAdjoint adjoint{
            .masses               = cuda::Buffer<float>(context.resource, configuration.rest_positions.size()),
            .stretch_stiffnesses  = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .stretch_dampings     = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .stretch_rest_lengths = cuda::Buffer<float>(context.resource, topology.stretch_springs.size()),
            .bending_stiffnesses  = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
            .bending_dampings     = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
            .bending_rest_lengths = cuda::Buffer<float>(context.resource, topology.bending_springs.size()),
        };
        if (adjoint.masses.size != 0u) context.resource->zero(adjoint.masses.data, adjoint.masses.size * sizeof(float));
        if (adjoint.stretch_stiffnesses.size != 0u) context.resource->zero(adjoint.stretch_stiffnesses.data, adjoint.stretch_stiffnesses.size * sizeof(float));
        if (adjoint.stretch_dampings.size != 0u) context.resource->zero(adjoint.stretch_dampings.data, adjoint.stretch_dampings.size * sizeof(float));
        if (adjoint.stretch_rest_lengths.size != 0u) context.resource->zero(adjoint.stretch_rest_lengths.data, adjoint.stretch_rest_lengths.size * sizeof(float));
        if (adjoint.bending_stiffnesses.size != 0u) context.resource->zero(adjoint.bending_stiffnesses.data, adjoint.bending_stiffnesses.size * sizeof(float));
        if (adjoint.bending_dampings.size != 0u) context.resource->zero(adjoint.bending_dampings.data, adjoint.bending_dampings.size * sizeof(float));
        if (adjoint.bending_rest_lengths.size != 0u) context.resource->zero(adjoint.bending_rest_lengths.data, adjoint.bending_rest_lengths.size * sizeof(float));
        return adjoint;
    }

    void Model::copy_state(const State& source, State& destination, ExecutionContext& context) const {
        context.resource->copy_device(destination.positions.x.data, source.positions.x.data, source.positions.x.size * sizeof(float));
        context.resource->copy_device(destination.positions.y.data, source.positions.y.data, source.positions.y.size * sizeof(float));
        context.resource->copy_device(destination.positions.z.data, source.positions.z.data, source.positions.z.size * sizeof(float));
        context.resource->copy_device(destination.velocities.x.data, source.velocities.x.data, source.velocities.x.size * sizeof(float));
        context.resource->copy_device(destination.velocities.y.data, source.velocities.y.data, source.velocities.y.size * sizeof(float));
        context.resource->copy_device(destination.velocities.z.data, source.velocities.z.data, source.velocities.z.size * sizeof(float));
    }

    void Model::copy_state_tangent(const StateTangent& source, StateTangent& destination, ExecutionContext& context) const {
        context.resource->copy_device(destination.positions.x.data, source.positions.x.data, source.positions.x.size * sizeof(float));
        context.resource->copy_device(destination.positions.y.data, source.positions.y.data, source.positions.y.size * sizeof(float));
        context.resource->copy_device(destination.positions.z.data, source.positions.z.data, source.positions.z.size * sizeof(float));
        context.resource->copy_device(destination.velocities.x.data, source.velocities.x.data, source.velocities.x.size * sizeof(float));
        context.resource->copy_device(destination.velocities.y.data, source.velocities.y.data, source.velocities.y.size * sizeof(float));
        context.resource->copy_device(destination.velocities.z.data, source.velocities.z.data, source.velocities.z.size * sizeof(float));
    }

    void Model::copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        context.resource->copy_device(destination.positions.x.data, source.positions.x.data, source.positions.x.size * sizeof(float));
        context.resource->copy_device(destination.positions.y.data, source.positions.y.data, source.positions.y.size * sizeof(float));
        context.resource->copy_device(destination.positions.z.data, source.positions.z.data, source.positions.z.size * sizeof(float));
        context.resource->copy_device(destination.velocities.x.data, source.velocities.x.data, source.velocities.x.size * sizeof(float));
        context.resource->copy_device(destination.velocities.y.data, source.velocities.y.data, source.velocities.y.size * sizeof(float));
        context.resource->copy_device(destination.velocities.z.data, source.velocities.z.data, source.velocities.z.size * sizeof(float));
    }

    void Model::accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        cuda_kernel::launch_accumulate(context.resource->native_stream, static_cast<std::uint32_t>(source.positions.x.size), {.x = source.positions.x.data, .y = source.positions.y.data, .z = source.positions.z.data}, {.x = destination.positions.x.data, .y = destination.positions.y.data, .z = destination.positions.z.data});
        cuda_kernel::launch_accumulate(context.resource->native_stream, static_cast<std::uint32_t>(source.velocities.x.size), {.x = source.velocities.x.data, .y = source.velocities.y.data, .z = source.velocities.z.data}, {.x = destination.velocities.x.data, .y = destination.velocities.y.data, .z = destination.velocities.z.data});
    }

    void Model::forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& step_cache, ExecutionContext& context) const {
        force_assembly_.forward(*context.resource, context.device_topology, configuration, state, control, parameters, step_cache.forces);
        semi_implicit_euler_.forward(*context.resource, configuration, state, parameters, step_cache.forces, context.integrated_state_);
        fixed_constraint_.forward(*context.resource, context.device_topology, context.integrated_state_, next_state);
    }

    void Model::jvp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const {
        force_assembly_.jvp(*context.resource, context.device_topology, configuration, state, parameters, state_tangent, control_tangent, parameter_tangent, context.force_tangent_);
        semi_implicit_euler_.jvp(*context.resource, configuration, parameters, step_cache.forces, state_tangent, parameter_tangent, context.force_tangent_, context.integrated_state_tangent_);
        fixed_constraint_.jvp(*context.resource, context.device_topology, context.integrated_state_tangent_, next_state_tangent);
    }

    void Model::vjp_step(const State& state, const Control&, const Parameters& parameters, const State&, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const {
        context.resource->zero(context.force_adjoint_.values.x.data, context.force_adjoint_.values.x.size * sizeof(float));
        context.resource->zero(context.force_adjoint_.values.y.data, context.force_adjoint_.values.y.size * sizeof(float));
        context.resource->zero(context.force_adjoint_.values.z.data, context.force_adjoint_.values.z.size * sizeof(float));
        context.resource->zero(context.integrated_state_adjoint_.positions.x.data, context.integrated_state_adjoint_.positions.x.size * sizeof(float));
        context.resource->zero(context.integrated_state_adjoint_.positions.y.data, context.integrated_state_adjoint_.positions.y.size * sizeof(float));
        context.resource->zero(context.integrated_state_adjoint_.positions.z.data, context.integrated_state_adjoint_.positions.z.size * sizeof(float));
        context.resource->zero(context.integrated_state_adjoint_.velocities.x.data, context.integrated_state_adjoint_.velocities.x.size * sizeof(float));
        context.resource->zero(context.integrated_state_adjoint_.velocities.y.data, context.integrated_state_adjoint_.velocities.y.size * sizeof(float));
        context.resource->zero(context.integrated_state_adjoint_.velocities.z.data, context.integrated_state_adjoint_.velocities.z.size * sizeof(float));
        fixed_constraint_.vjp(*context.resource, context.device_topology, next_state_adjoint, context.integrated_state_adjoint_);
        semi_implicit_euler_.vjp(*context.resource, configuration, parameters, step_cache.forces, context.integrated_state_adjoint_, previous_state_adjoint, context.force_adjoint_, parameter_adjoint);
        force_assembly_.vjp(*context.resource, context.device_topology, configuration, state, parameters, context.force_adjoint_, previous_state_adjoint, control_adjoint, parameter_adjoint);
    }

} // namespace xayah::cloth
