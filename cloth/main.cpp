#include <cuda_runtime_api.h>

import std;
import xayah.cloth.data;
import xayah.cloth.model;
import xayah.cloth.operators;
import xayah.solver;

namespace {

    constexpr float epsilon = 1.0e-2F;

    struct HostStateData {
        std::vector<xayah::cloth::Vector3> positions;
        std::vector<xayah::cloth::Vector3> velocities;
    };

    struct HostParameterData {
        std::vector<float> masses;
        std::vector<float> stretch_stiffnesses;
        std::vector<float> stretch_dampings;
        std::vector<float> stretch_rest_lengths;
        std::vector<float> bending_stiffnesses;
        std::vector<float> bending_dampings;
        std::vector<float> bending_rest_lengths;
    };

    struct DirectionMetrics {
        float finite_difference;
        float jvp_inner_product;
        float vjp_inner_product;
    };

    void check_cuda(const cudaError_t result) {
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    float relative_error(const float first, const float second) {
        return std::abs(first - second) / std::max({1.0e-8F, std::abs(first), std::abs(second)});
    }

    float dot(const xayah::cloth::Vector3 first, const xayah::cloth::Vector3 second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    float dot(const std::vector<xayah::cloth::Vector3>& first, const std::vector<xayah::cloth::Vector3>& second) {
        float result = 0.0F;
        for (std::size_t index = 0; index < first.size(); ++index) result += dot(first[index], second[index]);
        return result;
    }

    float dot(const std::vector<float>& first, const std::vector<float>& second) {
        return std::inner_product(first.begin(), first.end(), second.begin(), 0.0F);
    }

    float dot(const HostStateData& first, const HostStateData& second) {
        return dot(first.positions, second.positions) + dot(first.velocities, second.velocities);
    }

    float dot(const HostParameterData& first, const HostParameterData& second) {
        return dot(first.masses, second.masses) + dot(first.stretch_stiffnesses, second.stretch_stiffnesses) + dot(first.stretch_dampings, second.stretch_dampings) + dot(first.stretch_rest_lengths, second.stretch_rest_lengths) + dot(first.bending_stiffnesses, second.bending_stiffnesses) + dot(first.bending_dampings, second.bending_dampings) + dot(first.bending_rest_lengths, second.bending_rest_lengths);
    }

    std::vector<xayah::cloth::Vector3> add_scaled(const std::vector<xayah::cloth::Vector3>& values, const std::vector<xayah::cloth::Vector3>& direction, const float scale) {
        std::vector<xayah::cloth::Vector3> result(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) result[index] = {.x = values[index].x + scale * direction[index].x, .y = values[index].y + scale * direction[index].y, .z = values[index].z + scale * direction[index].z};
        return result;
    }

    std::vector<float> add_scaled(const std::vector<float>& values, const std::vector<float>& direction, const float scale) {
        std::vector<float> result(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) result[index] = values[index] + scale * direction[index];
        return result;
    }

    HostStateData add_scaled(const HostStateData& values, const HostStateData& direction, const float scale) {
        return {.positions = add_scaled(values.positions, direction.positions, scale), .velocities = add_scaled(values.velocities, direction.velocities, scale)};
    }

    HostParameterData add_scaled(const HostParameterData& values, const HostParameterData& direction, const float scale) {
        return {
            .masses               = add_scaled(values.masses, direction.masses, scale),
            .stretch_stiffnesses  = add_scaled(values.stretch_stiffnesses, direction.stretch_stiffnesses, scale),
            .stretch_dampings     = add_scaled(values.stretch_dampings, direction.stretch_dampings, scale),
            .stretch_rest_lengths = add_scaled(values.stretch_rest_lengths, direction.stretch_rest_lengths, scale),
            .bending_stiffnesses  = add_scaled(values.bending_stiffnesses, direction.bending_stiffnesses, scale),
            .bending_dampings     = add_scaled(values.bending_dampings, direction.bending_dampings, scale),
            .bending_rest_lengths = add_scaled(values.bending_rest_lengths, direction.bending_rest_lengths, scale),
        };
    }

    xayah::cloth::Configuration make_grid(const std::size_t resolution, const float extent, const float time_step) {
        xayah::cloth::Configuration configuration{
            .rest_positions = std::vector<xayah::cloth::Vector3>(resolution * resolution),
            .triangles      = {},
            .anchors        = std::vector<std::optional<xayah::cloth::Vector3>>(resolution * resolution),
            .gravity        = {.x = 0.0F, .y = -9.81F, .z = 0.0F},
            .time_step      = time_step,
        };
        for (std::size_t row = 0; row < resolution; ++row) {
            for (std::size_t column = 0; column < resolution; ++column) {
                const std::size_t particle             = row * resolution + column;
                configuration.rest_positions[particle] = {
                    .x = -extent * 0.5F + extent * static_cast<float>(column) / static_cast<float>(resolution - 1),
                    .y = 0.0F,
                    .z = extent * static_cast<float>(row) / static_cast<float>(resolution - 1),
                };
                if (row == 0) configuration.anchors[particle] = configuration.rest_positions[particle];
            }
        }
        configuration.triangles.reserve((resolution - 1) * (resolution - 1) * 2);
        for (std::size_t row = 0; row + 1 < resolution; ++row) {
            for (std::size_t column = 0; column + 1 < resolution; ++column) {
                const std::uint32_t top_left     = static_cast<std::uint32_t>(row * resolution + column);
                const std::uint32_t top_right    = top_left + 1;
                const std::uint32_t bottom_left  = top_left + static_cast<std::uint32_t>(resolution);
                const std::uint32_t bottom_right = bottom_left + 1;
                if ((row + column) % 2 == 0) {
                    configuration.triangles.push_back({.first = top_left, .second = bottom_left, .third = bottom_right});
                    configuration.triangles.push_back({.first = top_left, .second = bottom_right, .third = top_right});
                } else {
                    configuration.triangles.push_back({.first = top_left, .second = bottom_left, .third = top_right});
                    configuration.triangles.push_back({.first = top_right, .second = bottom_left, .third = bottom_right});
                }
            }
        }
        return configuration;
    }

    HostStateData make_state_data(const xayah::cloth::Configuration& configuration) {
        HostStateData state{.positions = configuration.rest_positions, .velocities = std::vector<xayah::cloth::Vector3>(configuration.rest_positions.size())};
        for (std::size_t particle = 0; particle < state.positions.size(); ++particle) {
            state.positions[particle].y += 0.015F * std::sin(0.37F * static_cast<float>(particle + 1));
            state.velocities[particle] = {.x = 0.02F * std::cos(0.19F * static_cast<float>(particle + 1)), .y = 0.01F * std::sin(0.23F * static_cast<float>(particle + 1)), .z = -0.015F * std::cos(0.29F * static_cast<float>(particle + 1))};
        }
        return state;
    }

    HostStateData make_state_direction(const std::size_t particle_count) {
        HostStateData direction{.positions = std::vector<xayah::cloth::Vector3>(particle_count), .velocities = std::vector<xayah::cloth::Vector3>(particle_count)};
        for (std::size_t particle = 0; particle < particle_count; ++particle) {
            const float index              = static_cast<float>(particle + 1);
            direction.positions[particle]  = {.x = 0.03F * std::sin(0.17F * index), .y = 0.03F * std::cos(0.31F * index), .z = 0.03F * std::sin(0.43F * index)};
            direction.velocities[particle] = {.x = 0.02F * std::cos(0.13F * index), .y = 0.02F * std::sin(0.27F * index), .z = 0.02F * std::cos(0.39F * index)};
        }
        return direction;
    }

    HostParameterData make_parameter_data(const xayah::cloth::Model& model, const float mass, const float stretch_stiffness, const float stretch_damping, const float bending_stiffness, const float bending_damping) {
        HostParameterData parameters{
            .masses               = std::vector<float>(model.configuration().rest_positions.size(), mass),
            .stretch_stiffnesses  = std::vector<float>(model.topology().stretch_springs.size(), stretch_stiffness),
            .stretch_dampings     = std::vector<float>(model.topology().stretch_springs.size(), stretch_damping),
            .stretch_rest_lengths = std::vector<float>(model.topology().stretch_springs.size()),
            .bending_stiffnesses  = std::vector<float>(model.topology().bending_springs.size(), bending_stiffness),
            .bending_dampings     = std::vector<float>(model.topology().bending_springs.size(), bending_damping),
            .bending_rest_lengths = std::vector<float>(model.topology().bending_springs.size()),
        };
        for (std::size_t spring = 0; spring < model.topology().stretch_springs.size(); ++spring) parameters.stretch_rest_lengths[spring] = model.topology().stretch_springs[spring].rest_length;
        for (std::size_t spring = 0; spring < model.topology().bending_springs.size(); ++spring) parameters.bending_rest_lengths[spring] = model.topology().bending_springs[spring].rest_length;
        return parameters;
    }

    HostParameterData make_parameter_direction(const xayah::cloth::Model& model) {
        HostParameterData direction{
            .masses               = std::vector<float>(model.configuration().rest_positions.size()),
            .stretch_stiffnesses  = std::vector<float>(model.topology().stretch_springs.size()),
            .stretch_dampings     = std::vector<float>(model.topology().stretch_springs.size()),
            .stretch_rest_lengths = std::vector<float>(model.topology().stretch_springs.size()),
            .bending_stiffnesses  = std::vector<float>(model.topology().bending_springs.size()),
            .bending_dampings     = std::vector<float>(model.topology().bending_springs.size()),
            .bending_rest_lengths = std::vector<float>(model.topology().bending_springs.size()),
        };
        for (std::size_t particle = 0; particle < direction.masses.size(); ++particle) direction.masses[particle] = 0.01F * std::sin(0.11F * static_cast<float>(particle + 1));
        for (std::size_t spring = 0; spring < direction.stretch_stiffnesses.size(); ++spring) {
            const float index                      = static_cast<float>(spring + 1);
            direction.stretch_stiffnesses[spring]  = 0.2F * std::sin(0.07F * index);
            direction.stretch_dampings[spring]     = 0.02F * std::cos(0.13F * index);
            direction.stretch_rest_lengths[spring] = 0.001F * std::sin(0.17F * index);
        }
        for (std::size_t spring = 0; spring < direction.bending_stiffnesses.size(); ++spring) {
            const float index                      = static_cast<float>(spring + 1);
            direction.bending_stiffnesses[spring]  = 0.05F * std::cos(0.09F * index);
            direction.bending_dampings[spring]     = 0.01F * std::sin(0.15F * index);
            direction.bending_rest_lengths[spring] = 0.001F * std::cos(0.21F * index);
        }
        return direction;
    }

    std::vector<xayah::cloth::Vector3> make_field(const std::size_t particle_count, const float scale, const float phase) {
        std::vector<xayah::cloth::Vector3> field(particle_count);
        for (std::size_t particle = 0; particle < particle_count; ++particle) {
            const float index = static_cast<float>(particle + 1);
            field[particle]   = {.x = scale * std::sin(0.17F * index + phase), .y = scale * std::cos(0.23F * index + phase), .z = scale * std::sin(0.31F * index - phase)};
        }
        return field;
    }

    void upload(xayah::cloth::ExecutionContext& context, const HostStateData& source, xayah::cloth::State& destination) {
        context.upload(source.positions, destination.positions);
        context.upload(source.velocities, destination.velocities);
    }

    void upload(xayah::cloth::ExecutionContext& context, const HostStateData& source, xayah::cloth::StateTangent& destination) {
        context.upload(source.positions, destination.positions);
        context.upload(source.velocities, destination.velocities);
    }

    void upload(xayah::cloth::ExecutionContext& context, const HostStateData& source, xayah::cloth::StateAdjoint& destination) {
        context.upload(source.positions, destination.positions);
        context.upload(source.velocities, destination.velocities);
    }

    void upload(xayah::cloth::ExecutionContext& context, const HostParameterData& source, xayah::cloth::Parameters& destination) {
        context.upload(source.masses, destination.masses);
        context.upload(source.stretch_stiffnesses, destination.stretch_stiffnesses);
        context.upload(source.stretch_dampings, destination.stretch_dampings);
        context.upload(source.stretch_rest_lengths, destination.stretch_rest_lengths);
        context.upload(source.bending_stiffnesses, destination.bending_stiffnesses);
        context.upload(source.bending_dampings, destination.bending_dampings);
        context.upload(source.bending_rest_lengths, destination.bending_rest_lengths);
    }

    void upload(xayah::cloth::ExecutionContext& context, const HostParameterData& source, xayah::cloth::ParameterTangent& destination) {
        context.upload(source.masses, destination.masses);
        context.upload(source.stretch_stiffnesses, destination.stretch_stiffnesses);
        context.upload(source.stretch_dampings, destination.stretch_dampings);
        context.upload(source.stretch_rest_lengths, destination.stretch_rest_lengths);
        context.upload(source.bending_stiffnesses, destination.bending_stiffnesses);
        context.upload(source.bending_dampings, destination.bending_dampings);
        context.upload(source.bending_rest_lengths, destination.bending_rest_lengths);
    }

    HostStateData download(xayah::cloth::ExecutionContext& context, const xayah::cloth::State& source) {
        HostStateData result{.positions = std::vector<xayah::cloth::Vector3>(source.positions.x.size()), .velocities = std::vector<xayah::cloth::Vector3>(source.velocities.x.size())};
        context.download(source.positions, result.positions);
        context.download(source.velocities, result.velocities);
        return result;
    }

    HostStateData download(xayah::cloth::ExecutionContext& context, const xayah::cloth::StateTangent& source) {
        HostStateData result{.positions = std::vector<xayah::cloth::Vector3>(source.positions.x.size()), .velocities = std::vector<xayah::cloth::Vector3>(source.velocities.x.size())};
        context.download(source.positions, result.positions);
        context.download(source.velocities, result.velocities);
        return result;
    }

    HostStateData download(xayah::cloth::ExecutionContext& context, const xayah::cloth::StateAdjoint& source) {
        HostStateData result{.positions = std::vector<xayah::cloth::Vector3>(source.positions.x.size()), .velocities = std::vector<xayah::cloth::Vector3>(source.velocities.x.size())};
        context.download(source.positions, result.positions);
        context.download(source.velocities, result.velocities);
        return result;
    }

    HostParameterData download(xayah::cloth::ExecutionContext& context, const xayah::cloth::ParameterAdjoint& source) {
        HostParameterData result{
            .masses               = std::vector<float>(source.masses.size()),
            .stretch_stiffnesses  = std::vector<float>(source.stretch_stiffnesses.size()),
            .stretch_dampings     = std::vector<float>(source.stretch_dampings.size()),
            .stretch_rest_lengths = std::vector<float>(source.stretch_rest_lengths.size()),
            .bending_stiffnesses  = std::vector<float>(source.bending_stiffnesses.size()),
            .bending_dampings     = std::vector<float>(source.bending_dampings.size()),
            .bending_rest_lengths = std::vector<float>(source.bending_rest_lengths.size()),
        };
        context.download(source.masses, result.masses);
        context.download(source.stretch_stiffnesses, result.stretch_stiffnesses);
        context.download(source.stretch_dampings, result.stretch_dampings);
        context.download(source.stretch_rest_lengths, result.stretch_rest_lengths);
        context.download(source.bending_stiffnesses, result.bending_stiffnesses);
        context.download(source.bending_dampings, result.bending_dampings);
        context.download(source.bending_rest_lengths, result.bending_rest_lengths);
        return result;
    }

    void print_metrics(const std::string_view name, const DirectionMetrics metrics) {
        std::println("  {}: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", name, metrics.finite_difference, metrics.jvp_inner_product, metrics.vjp_inner_product, relative_error(metrics.finite_difference, metrics.jvp_inner_product), relative_error(metrics.jvp_inner_product, metrics.vjp_inner_product));
    }

    DirectionMetrics check_force_operator() {
        xayah::cloth::Model model(make_grid(4, 1.0F, 1.0F / 240.0F));
        xayah::cloth::ExecutionContext context                     = model.make_context();
        const HostStateData state_data                             = make_state_data(model.configuration());
        const HostStateData state_direction                        = make_state_direction(state_data.positions.size());
        const HostParameterData parameter_data                     = make_parameter_data(model, 0.2F, 40.0F, 0.4F, 4.0F, 0.1F);
        const HostParameterData parameter_direction                = make_parameter_direction(model);
        const std::vector<xayah::cloth::Vector3> control_data      = make_field(state_data.positions.size(), 0.05F, 0.2F);
        const std::vector<xayah::cloth::Vector3> control_direction = make_field(state_data.positions.size(), 0.03F, 0.7F);
        const std::vector<xayah::cloth::Vector3> output_seed       = make_field(state_data.positions.size(), 0.5F, 1.1F);

        xayah::cloth::State state = model.make_state(context);
        upload(context, state_data, state);
        xayah::cloth::Parameters parameters = model.make_parameters(context);
        upload(context, parameter_data, parameters);
        xayah::cloth::Control control = model.make_control(context);
        context.upload(control_data, control.external_forces);
        xayah::cloth::StateTangent state_tangent = model.make_state_tangent(context);
        upload(context, state_direction, state_tangent);
        xayah::cloth::ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        upload(context, parameter_direction, parameter_tangent);
        xayah::cloth::ControlTangent control_tangent = model.make_control_tangent(context);
        context.upload(control_direction, control_tangent.external_forces);

        xayah::cloth::ForceTangent force_tangent = model.make_force_tangent(context);
        xayah::cloth::ForceAssemblyOperator force_operator;
        force_operator.jvp(context.resource(), context.device_topology(), model.configuration(), state, parameters, state_tangent, control_tangent, parameter_tangent, force_tangent);
        std::vector<xayah::cloth::Vector3> force_tangent_data(state_data.positions.size());
        context.download(force_tangent.values, force_tangent_data);

        xayah::cloth::ForceAdjoint force_adjoint = model.make_force_adjoint(context);
        context.upload(output_seed, force_adjoint.values);
        xayah::cloth::StateAdjoint state_adjoint         = model.make_state_adjoint(context);
        xayah::cloth::ControlAdjoint control_adjoint     = model.make_control_adjoint(context);
        xayah::cloth::ParameterAdjoint parameter_adjoint = model.make_parameter_adjoint(context);
        force_operator.vjp(context.resource(), context.device_topology(), model.configuration(), state, parameters, force_adjoint, state_adjoint, control_adjoint, parameter_adjoint);
        const HostStateData state_adjoint_data = download(context, state_adjoint);
        std::vector<xayah::cloth::Vector3> control_adjoint_data(control_data.size());
        context.download(control_adjoint.external_forces, control_adjoint_data);
        const HostParameterData parameter_adjoint_data = download(context, parameter_adjoint);

        const HostStateData plus_state_data          = add_scaled(state_data, state_direction, epsilon);
        const HostStateData minus_state_data         = add_scaled(state_data, state_direction, -epsilon);
        const HostParameterData plus_parameter_data  = add_scaled(parameter_data, parameter_direction, epsilon);
        const HostParameterData minus_parameter_data = add_scaled(parameter_data, parameter_direction, -epsilon);
        xayah::cloth::State plus_state               = model.make_state(context);
        xayah::cloth::State minus_state              = model.make_state(context);
        upload(context, plus_state_data, plus_state);
        upload(context, minus_state_data, minus_state);
        xayah::cloth::Parameters plus_parameters  = model.make_parameters(context);
        xayah::cloth::Parameters minus_parameters = model.make_parameters(context);
        upload(context, plus_parameter_data, plus_parameters);
        upload(context, minus_parameter_data, minus_parameters);
        xayah::cloth::Control plus_control  = model.make_control(context);
        xayah::cloth::Control minus_control = model.make_control(context);
        context.upload(add_scaled(control_data, control_direction, epsilon), plus_control.external_forces);
        context.upload(add_scaled(control_data, control_direction, -epsilon), minus_control.external_forces);
        xayah::cloth::Forces plus_forces  = model.make_forces(context);
        xayah::cloth::Forces minus_forces = model.make_forces(context);
        force_operator.forward(context.resource(), context.device_topology(), model.configuration(), plus_state, plus_control, plus_parameters, plus_forces);
        force_operator.forward(context.resource(), context.device_topology(), model.configuration(), minus_state, minus_control, minus_parameters, minus_forces);
        std::vector<xayah::cloth::Vector3> plus_force_data(output_seed.size());
        std::vector<xayah::cloth::Vector3> minus_force_data(output_seed.size());
        context.download(plus_forces.values, plus_force_data);
        context.download(minus_forces.values, minus_force_data);

        return {
            .finite_difference = (dot(output_seed, plus_force_data) - dot(output_seed, minus_force_data)) / (2.0F * epsilon),
            .jvp_inner_product = dot(output_seed, force_tangent_data),
            .vjp_inner_product = dot(state_direction, state_adjoint_data) + dot(control_direction, control_adjoint_data) + dot(parameter_direction, parameter_adjoint_data),
        };
    }

    DirectionMetrics check_euler_operator() {
        xayah::cloth::Model model(make_grid(4, 1.0F, 1.0F / 240.0F));
        xayah::cloth::ExecutionContext context = model.make_context();
        const HostStateData state_data         = make_state_data(model.configuration());
        const HostStateData state_direction    = make_state_direction(state_data.positions.size());
        const HostParameterData parameter_data = make_parameter_data(model, 0.2F, 40.0F, 0.4F, 4.0F, 0.1F);
        HostParameterData parameter_direction  = make_parameter_direction(model);
        std::ranges::fill(parameter_direction.stretch_stiffnesses, 0.0F);
        std::ranges::fill(parameter_direction.stretch_dampings, 0.0F);
        std::ranges::fill(parameter_direction.stretch_rest_lengths, 0.0F);
        std::ranges::fill(parameter_direction.bending_stiffnesses, 0.0F);
        std::ranges::fill(parameter_direction.bending_dampings, 0.0F);
        std::ranges::fill(parameter_direction.bending_rest_lengths, 0.0F);
        const std::vector<xayah::cloth::Vector3> force_data      = make_field(state_data.positions.size(), 0.4F, 0.4F);
        const std::vector<xayah::cloth::Vector3> force_direction = make_field(state_data.positions.size(), 0.1F, 0.9F);
        const HostStateData output_seed{.positions = make_field(state_data.positions.size(), 0.4F, 1.2F), .velocities = make_field(state_data.positions.size(), 0.3F, 1.7F)};

        xayah::cloth::State state = model.make_state(context);
        upload(context, state_data, state);
        xayah::cloth::Parameters parameters = model.make_parameters(context);
        upload(context, parameter_data, parameters);
        xayah::cloth::Forces forces = model.make_forces(context);
        context.upload(force_data, forces.values);
        xayah::cloth::StateTangent state_tangent = model.make_state_tangent(context);
        upload(context, state_direction, state_tangent);
        xayah::cloth::ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        upload(context, parameter_direction, parameter_tangent);
        xayah::cloth::ForceTangent force_tangent = model.make_force_tangent(context);
        context.upload(force_direction, force_tangent.values);

        xayah::cloth::SemiImplicitEulerOperator euler_operator;
        xayah::cloth::StateTangent output_tangent = model.make_state_tangent(context);
        euler_operator.jvp(context.resource(), model.configuration(), parameters, forces, state_tangent, parameter_tangent, force_tangent, output_tangent);
        const HostStateData output_tangent_data = download(context, output_tangent);

        xayah::cloth::StateAdjoint output_adjoint = model.make_state_adjoint(context);
        upload(context, output_seed, output_adjoint);
        xayah::cloth::StateAdjoint state_adjoint         = model.make_state_adjoint(context);
        xayah::cloth::ForceAdjoint force_adjoint         = model.make_force_adjoint(context);
        xayah::cloth::ParameterAdjoint parameter_adjoint = model.make_parameter_adjoint(context);
        euler_operator.vjp(context.resource(), model.configuration(), parameters, forces, output_adjoint, state_adjoint, force_adjoint, parameter_adjoint);
        const HostStateData state_adjoint_data = download(context, state_adjoint);
        std::vector<xayah::cloth::Vector3> force_adjoint_data(force_data.size());
        context.download(force_adjoint.values, force_adjoint_data);
        const HostParameterData parameter_adjoint_data = download(context, parameter_adjoint);

        xayah::cloth::State plus_state  = model.make_state(context);
        xayah::cloth::State minus_state = model.make_state(context);
        upload(context, add_scaled(state_data, state_direction, epsilon), plus_state);
        upload(context, add_scaled(state_data, state_direction, -epsilon), minus_state);
        xayah::cloth::Parameters plus_parameters  = model.make_parameters(context);
        xayah::cloth::Parameters minus_parameters = model.make_parameters(context);
        upload(context, add_scaled(parameter_data, parameter_direction, epsilon), plus_parameters);
        upload(context, add_scaled(parameter_data, parameter_direction, -epsilon), minus_parameters);
        xayah::cloth::Forces plus_forces  = model.make_forces(context);
        xayah::cloth::Forces minus_forces = model.make_forces(context);
        context.upload(add_scaled(force_data, force_direction, epsilon), plus_forces.values);
        context.upload(add_scaled(force_data, force_direction, -epsilon), minus_forces.values);
        xayah::cloth::State plus_output  = model.make_state(context);
        xayah::cloth::State minus_output = model.make_state(context);
        euler_operator.forward(context.resource(), model.configuration(), plus_state, plus_parameters, plus_forces, plus_output);
        euler_operator.forward(context.resource(), model.configuration(), minus_state, minus_parameters, minus_forces, minus_output);

        return {
            .finite_difference = (dot(output_seed, download(context, plus_output)) - dot(output_seed, download(context, minus_output))) / (2.0F * epsilon),
            .jvp_inner_product = dot(output_seed, output_tangent_data),
            .vjp_inner_product = dot(state_direction, state_adjoint_data) + dot(force_direction, force_adjoint_data) + dot(parameter_direction, parameter_adjoint_data),
        };
    }

    DirectionMetrics check_constraint_operator() {
        xayah::cloth::Model model(make_grid(4, 1.0F, 1.0F / 240.0F));
        xayah::cloth::ExecutionContext context = model.make_context();
        const HostStateData state_data         = make_state_data(model.configuration());
        const HostStateData state_direction    = make_state_direction(state_data.positions.size());
        const HostStateData output_seed{.positions = make_field(state_data.positions.size(), 0.4F, 0.8F), .velocities = make_field(state_data.positions.size(), 0.3F, 1.3F)};
        xayah::cloth::FixedConstraintOperator constraint_operator;

        xayah::cloth::StateTangent state_tangent = model.make_state_tangent(context);
        upload(context, state_direction, state_tangent);
        xayah::cloth::StateTangent output_tangent = model.make_state_tangent(context);
        constraint_operator.jvp(context.resource(), context.device_topology(), state_tangent, output_tangent);
        const HostStateData output_tangent_data = download(context, output_tangent);

        xayah::cloth::StateAdjoint output_adjoint = model.make_state_adjoint(context);
        upload(context, output_seed, output_adjoint);
        xayah::cloth::StateAdjoint state_adjoint = model.make_state_adjoint(context);
        constraint_operator.vjp(context.resource(), context.device_topology(), output_adjoint, state_adjoint);
        const HostStateData state_adjoint_data = download(context, state_adjoint);

        xayah::cloth::State plus_state  = model.make_state(context);
        xayah::cloth::State minus_state = model.make_state(context);
        upload(context, add_scaled(state_data, state_direction, epsilon), plus_state);
        upload(context, add_scaled(state_data, state_direction, -epsilon), minus_state);
        xayah::cloth::State plus_output  = model.make_state(context);
        xayah::cloth::State minus_output = model.make_state(context);
        constraint_operator.forward(context.resource(), context.device_topology(), plus_state, plus_output);
        constraint_operator.forward(context.resource(), context.device_topology(), minus_state, minus_output);

        return {
            .finite_difference = (dot(output_seed, download(context, plus_output)) - dot(output_seed, download(context, minus_output))) / (2.0F * epsilon),
            .jvp_inner_product = dot(output_seed, output_tangent_data),
            .vjp_inner_product = dot(state_direction, state_adjoint_data),
        };
    }

    float trajectory_loss(xayah::cloth::ExecutionContext& context, const xayah::solver::Trajectory<xayah::cloth::State, xayah::cloth::StepCache>& trajectory, const xayah::cloth::Configuration& configuration) {
        float loss                = 0.0F;
        const float normalization = 1.0F / static_cast<float>(trajectory.states.size() * configuration.rest_positions.size());
        for (std::size_t step = 0; step < trajectory.states.size(); ++step) {
            const HostStateData state = download(context, trajectory.states[step]);
            for (std::size_t particle = 0; particle < state.positions.size(); ++particle) {
                const float target_y = configuration.rest_positions[particle].y - 0.05F * static_cast<float>(step) / static_cast<float>(trajectory.states.size() - 1);
                const xayah::cloth::Vector3 residual{.x = state.positions[particle].x - configuration.rest_positions[particle].x, .y = state.positions[particle].y - target_y, .z = state.positions[particle].z - configuration.rest_positions[particle].z};
                loss += 0.5F * normalization * (dot(residual, residual) + 0.1F * dot(state.velocities[particle], state.velocities[particle]));
            }
        }
        return loss;
    }

    xayah::solver::TrajectoryAdjoint<xayah::cloth::StateAdjoint> make_trajectory_seeds(xayah::cloth::Model& model, xayah::cloth::ExecutionContext& context, const xayah::solver::Trajectory<xayah::cloth::State, xayah::cloth::StepCache>& trajectory) {
        xayah::solver::TrajectoryAdjoint<xayah::cloth::StateAdjoint> seeds{.states = {}};
        seeds.states.reserve(trajectory.states.size());
        const float normalization = 1.0F / static_cast<float>(trajectory.states.size() * model.configuration().rest_positions.size());
        for (std::size_t step = 0; step < trajectory.states.size(); ++step) {
            const HostStateData state = download(context, trajectory.states[step]);
            HostStateData seed{.positions = std::vector<xayah::cloth::Vector3>(state.positions.size()), .velocities = std::vector<xayah::cloth::Vector3>(state.velocities.size())};
            for (std::size_t particle = 0; particle < state.positions.size(); ++particle) {
                const float target_y     = model.configuration().rest_positions[particle].y - 0.05F * static_cast<float>(step) / static_cast<float>(trajectory.states.size() - 1);
                seed.positions[particle] = {
                    .x = normalization * (state.positions[particle].x - model.configuration().rest_positions[particle].x),
                    .y = normalization * (state.positions[particle].y - target_y),
                    .z = normalization * (state.positions[particle].z - model.configuration().rest_positions[particle].z),
                };
                seed.velocities[particle] = {.x = 0.1F * normalization * state.velocities[particle].x, .y = 0.1F * normalization * state.velocities[particle].y, .z = 0.1F * normalization * state.velocities[particle].z};
            }
            xayah::cloth::StateAdjoint state_seed = model.make_state_adjoint(context);
            upload(context, seed, state_seed);
            seeds.states.push_back(std::move(state_seed));
        }
        return seeds;
    }

    std::vector<xayah::cloth::Control> make_controls(xayah::cloth::Model& model, xayah::cloth::ExecutionContext& context, const std::vector<std::vector<xayah::cloth::Vector3>>& values) {
        std::vector<xayah::cloth::Control> controls;
        controls.reserve(values.size());
        for (const std::vector<xayah::cloth::Vector3>& value : values) {
            xayah::cloth::Control control = model.make_control(context);
            context.upload(value, control.external_forces);
            controls.push_back(std::move(control));
        }
        return controls;
    }

    std::vector<xayah::cloth::ControlTangent> make_control_tangents(xayah::cloth::Model& model, xayah::cloth::ExecutionContext& context, const std::vector<std::vector<xayah::cloth::Vector3>>& values) {
        std::vector<xayah::cloth::ControlTangent> tangents;
        tangents.reserve(values.size());
        for (const std::vector<xayah::cloth::Vector3>& value : values) {
            xayah::cloth::ControlTangent tangent = model.make_control_tangent(context);
            context.upload(value, tangent.external_forces);
            tangents.push_back(std::move(tangent));
        }
        return tangents;
    }

    float trajectory_jvp_inner(xayah::cloth::ExecutionContext& context, const xayah::solver::Trajectory<xayah::cloth::State, xayah::cloth::StepCache>& trajectory, const xayah::solver::TrajectoryTangent<xayah::cloth::StateTangent>& tangent, const xayah::cloth::Configuration& configuration) {
        float result              = 0.0F;
        const float normalization = 1.0F / static_cast<float>(trajectory.states.size() * configuration.rest_positions.size());
        for (std::size_t step = 0; step < trajectory.states.size(); ++step) {
            const HostStateData state     = download(context, trajectory.states[step]);
            const HostStateData direction = download(context, tangent.states[step]);
            for (std::size_t particle = 0; particle < state.positions.size(); ++particle) {
                const float target_y = configuration.rest_positions[particle].y - 0.05F * static_cast<float>(step) / static_cast<float>(trajectory.states.size() - 1);
                const xayah::cloth::Vector3 residual{.x = state.positions[particle].x - configuration.rest_positions[particle].x, .y = state.positions[particle].y - target_y, .z = state.positions[particle].z - configuration.rest_positions[particle].z};
                result += normalization * (dot(residual, direction.positions[particle]) + 0.1F * dot(state.velocities[particle], direction.velocities[particle]));
            }
        }
        return result;
    }

    DirectionMetrics check_trajectory(const xayah::solver::TapeMode tape_mode, const int direction_block) {
        constexpr std::size_t step_count = 20;
        xayah::cloth::Model model(make_grid(8, 1.5F, 1.0F / 240.0F));
        xayah::cloth::ExecutionContext context = model.make_context();
        const HostStateData state_data         = make_state_data(model.configuration());
        HostStateData state_direction          = make_state_direction(state_data.positions.size());
        const HostParameterData parameter_data = make_parameter_data(model, 0.1F, 100.0F, 0.8F, 8.0F, 0.15F);
        HostParameterData parameter_direction  = make_parameter_direction(model);
        std::vector<std::vector<xayah::cloth::Vector3>> control_data(step_count, std::vector<xayah::cloth::Vector3>(state_data.positions.size()));
        std::vector<std::vector<xayah::cloth::Vector3>> control_direction(step_count);
        for (std::size_t step = 0; step < step_count; ++step) control_direction[step] = make_field(state_data.positions.size(), 1.0F, 0.1F * static_cast<float>(step + 1));

        if (direction_block != 0) state_direction = {.positions = std::vector<xayah::cloth::Vector3>(state_data.positions.size()), .velocities = std::vector<xayah::cloth::Vector3>(state_data.positions.size())};
        if (direction_block != 1) control_direction.assign(step_count, std::vector<xayah::cloth::Vector3>(state_data.positions.size()));
        if (direction_block == 2) {
            for (float& value : parameter_direction.masses) value *= 50.0F;
            for (float& value : parameter_direction.stretch_stiffnesses) value *= 50.0F;
            for (float& value : parameter_direction.stretch_dampings) value *= 50.0F;
            for (float& value : parameter_direction.stretch_rest_lengths) value *= 50.0F;
            for (float& value : parameter_direction.bending_stiffnesses) value *= 50.0F;
            for (float& value : parameter_direction.bending_dampings) value *= 50.0F;
            for (float& value : parameter_direction.bending_rest_lengths) value *= 50.0F;
        }
        if (direction_block != 2)
            parameter_direction = {
                .masses               = std::vector<float>(parameter_data.masses.size()),
                .stretch_stiffnesses  = std::vector<float>(parameter_data.stretch_stiffnesses.size()),
                .stretch_dampings     = std::vector<float>(parameter_data.stretch_dampings.size()),
                .stretch_rest_lengths = std::vector<float>(parameter_data.stretch_rest_lengths.size()),
                .bending_stiffnesses  = std::vector<float>(parameter_data.bending_stiffnesses.size()),
                .bending_dampings     = std::vector<float>(parameter_data.bending_dampings.size()),
                .bending_rest_lengths = std::vector<float>(parameter_data.bending_rest_lengths.size()),
            };

        xayah::cloth::State state = model.make_state(context);
        upload(context, state_data, state);
        xayah::cloth::Parameters parameters = model.make_parameters(context);
        upload(context, parameter_data, parameters);
        std::vector<xayah::cloth::Control> controls = make_controls(model, context, control_data);
        const auto trajectory                       = xayah::solver::simulate(model, context, state, std::span<const xayah::cloth::Control>{controls}, parameters, tape_mode);
        const auto seeds                            = make_trajectory_seeds(model, context, trajectory);

        xayah::cloth::StateTangent state_tangent = model.make_state_tangent(context);
        upload(context, state_direction, state_tangent);
        xayah::cloth::ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        upload(context, parameter_direction, parameter_tangent);
        std::vector<xayah::cloth::ControlTangent> control_tangents = make_control_tangents(model, context, control_direction);
        const auto trajectory_tangent                              = xayah::solver::jvp(model, context, trajectory, std::span<const xayah::cloth::Control>{controls}, parameters, state_tangent, std::span<const xayah::cloth::ControlTangent>{control_tangents}, parameter_tangent);
        const auto gradients                                       = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::cloth::Control>{controls}, parameters, seeds);

        float vjp_inner_product = dot(state_direction, download(context, gradients.initial_state)) + dot(parameter_direction, download(context, gradients.parameters));
        for (std::size_t step = 0; step < step_count; ++step) {
            std::vector<xayah::cloth::Vector3> gradient(state_data.positions.size());
            context.download(gradients.controls[step].external_forces, gradient);
            vjp_inner_product += dot(control_direction[step], gradient);
        }

        xayah::cloth::State plus_state  = model.make_state(context);
        xayah::cloth::State minus_state = model.make_state(context);
        upload(context, add_scaled(state_data, state_direction, epsilon), plus_state);
        upload(context, add_scaled(state_data, state_direction, -epsilon), minus_state);
        xayah::cloth::Parameters plus_parameters  = model.make_parameters(context);
        xayah::cloth::Parameters minus_parameters = model.make_parameters(context);
        upload(context, add_scaled(parameter_data, parameter_direction, epsilon), plus_parameters);
        upload(context, add_scaled(parameter_data, parameter_direction, -epsilon), minus_parameters);
        std::vector<std::vector<xayah::cloth::Vector3>> plus_control_data(step_count);
        std::vector<std::vector<xayah::cloth::Vector3>> minus_control_data(step_count);
        for (std::size_t step = 0; step < step_count; ++step) {
            plus_control_data[step]  = add_scaled(control_data[step], control_direction[step], epsilon);
            minus_control_data[step] = add_scaled(control_data[step], control_direction[step], -epsilon);
        }
        std::vector<xayah::cloth::Control> plus_controls  = make_controls(model, context, plus_control_data);
        std::vector<xayah::cloth::Control> minus_controls = make_controls(model, context, minus_control_data);
        const auto plus_trajectory                        = xayah::solver::simulate(model, context, plus_state, std::span<const xayah::cloth::Control>{plus_controls}, plus_parameters, tape_mode);
        const auto minus_trajectory                       = xayah::solver::simulate(model, context, minus_state, std::span<const xayah::cloth::Control>{minus_controls}, minus_parameters, tape_mode);

        return {
            .finite_difference = (trajectory_loss(context, plus_trajectory, model.configuration()) - trajectory_loss(context, minus_trajectory, model.configuration())) / (2.0F * epsilon),
            .jvp_inner_product = trajectory_jvp_inner(context, trajectory, trajectory_tangent, model.configuration()),
            .vjp_inner_product = vjp_inner_product,
        };
    }

    void run_hanging_demo() {
        constexpr std::size_t resolution = 32;
        constexpr std::size_t step_count = 240;
        xayah::cloth::Model model(make_grid(resolution, 2.0F, 1.0F / 240.0F));
        xayah::cloth::ExecutionContext context = model.make_context();
        xayah::cloth::State initial_state      = model.make_state(context);
        context.upload(model.configuration().rest_positions, initial_state.positions);
        context.upload(std::vector<xayah::cloth::Vector3>(model.configuration().rest_positions.size()), initial_state.velocities);
        xayah::cloth::Parameters parameters    = model.make_parameters(context);
        const HostParameterData parameter_data = make_parameter_data(model, 0.05F, 400.0F, 1.0F, 5.0F, 0.1F);
        upload(context, parameter_data, parameters);
        std::vector<std::vector<xayah::cloth::Vector3>> control_data(step_count, std::vector<xayah::cloth::Vector3>(model.configuration().rest_positions.size()));
        std::vector<xayah::cloth::Control> controls = make_controls(model, context, control_data);

        cudaEvent_t start;
        cudaEvent_t finish;
        check_cuda(cudaEventCreate(&start));
        check_cuda(cudaEventCreate(&finish));
        check_cuda(cudaEventRecord(start, static_cast<cudaStream_t>(context.resource().native_stream())));
        const auto trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::cloth::Control>{controls}, parameters, xayah::solver::TapeMode::recompute_step_cache);
        check_cuda(cudaEventRecord(finish, static_cast<cudaStream_t>(context.resource().native_stream())));
        check_cuda(cudaEventSynchronize(finish));
        float elapsed_milliseconds;
        check_cuda(cudaEventElapsedTime(&elapsed_milliseconds, start, finish));
        check_cuda(cudaEventDestroy(start));
        check_cuda(cudaEventDestroy(finish));

        const HostStateData final_state = download(context, trajectory.states.back());
        float minimum_y                 = final_state.positions.front().y;
        float maximum_speed             = 0.0F;
        for (std::size_t particle = 0; particle < final_state.positions.size(); ++particle) {
            minimum_y     = std::min(minimum_y, final_state.positions[particle].y);
            maximum_speed = std::max(maximum_speed, std::sqrt(dot(final_state.velocities[particle], final_state.velocities[particle])));
        }
        const std::size_t center = (resolution / 2) * resolution + resolution / 2;

        std::ofstream output("cloth-final.obj");
        for (const xayah::cloth::Vector3 position : final_state.positions) output << "v " << position.x << ' ' << position.y << ' ' << position.z << '\n';
        for (const xayah::cloth::Triangle triangle : model.configuration().triangles) output << "f " << triangle.first + 1 << ' ' << triangle.second + 1 << ' ' << triangle.third + 1 << '\n';

        std::println("\nHanging cloth 32x32");
        std::println("  center=({:.6f}, {:.6f}, {:.6f})", final_state.positions[center].x, final_state.positions[center].y, final_state.positions[center].z);
        std::println("  minimum_y={:.6f}, maximum_speed={:.6f}", minimum_y, maximum_speed);
        std::println("  CUDA elapsed={:.3f} ms, output=cloth-final.obj", elapsed_milliseconds);
    }

} // namespace

int main() {
    std::println("CUDA cloth operator gradient checks");
    print_metrics("ForceAssembly", check_force_operator());
    print_metrics("SemiImplicitEuler", check_euler_operator());
    print_metrics("FixedConstraint", check_constraint_operator());

    std::println("\nCUDA cloth trajectory gradient checks");
    std::array<DirectionMetrics, 3> store_all_metrics;
    std::array<DirectionMetrics, 3> recompute_metrics;
    for (int block = 0; block < 3; ++block) store_all_metrics[block] = check_trajectory(xayah::solver::TapeMode::store_all, block);
    for (int block = 0; block < 3; ++block) recompute_metrics[block] = check_trajectory(xayah::solver::TapeMode::recompute_step_cache, block);
    const std::array<std::string_view, 3> block_names{"initial state", "controls", "parameters"};
    for (int block = 0; block < 3; ++block) print_metrics(std::format("store_all / {}", block_names[block]), store_all_metrics[block]);
    for (int block = 0; block < 3; ++block) print_metrics(std::format("recompute_step_cache / {}", block_names[block]), recompute_metrics[block]);
    for (int block = 0; block < 3; ++block) {
        const float tape_difference = std::max({std::abs(store_all_metrics[block].finite_difference - recompute_metrics[block].finite_difference), std::abs(store_all_metrics[block].jvp_inner_product - recompute_metrics[block].jvp_inner_product), std::abs(store_all_metrics[block].vjp_inner_product - recompute_metrics[block].vjp_inner_product)});
        std::println("  tape consistency / {}: max_abs_difference={:.9e}", block_names[block], tape_difference);
    }

    run_hanging_demo();
    return 0;
}
