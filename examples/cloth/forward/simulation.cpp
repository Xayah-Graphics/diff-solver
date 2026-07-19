module;

#include "simulation.h"
#include <cuda_runtime_api.h>

module xayah.examples.cloth.forward;

import std;
import xayah.cloth.data;
import xayah.cloth.model;

namespace xayah::cloth::examples::forward {

    namespace {

        Configuration make_configuration(const ForwardSimulationOptions& options) {
            Configuration configuration{
                .rest_positions = std::vector<Vector3>(static_cast<std::size_t>(options.rows) * options.columns),
                .triangles = {},
                .anchors = std::vector<std::optional<Vector3>>(static_cast<std::size_t>(options.rows) * options.columns),
                .gravity = {.x = 0.0F, .y = options.gravity_y, .z = 0.0F},
                .time_step = options.time_step,
            };
            const float spacing_x = options.width / static_cast<float>(options.columns - 1u);
            const float spacing_y = options.height / static_cast<float>(options.rows - 1u);
            for (std::uint32_t row = 0u; row < options.rows; ++row) {
                for (std::uint32_t column = 0u; column < options.columns; ++column) {
                    const std::uint32_t particle = row * options.columns + column;
                    configuration.rest_positions[particle] = {.x = static_cast<float>(column) * spacing_x, .y = -static_cast<float>(row) * spacing_y, .z = 0.0F};
                    if (column == 0u) configuration.anchors[particle] = configuration.rest_positions[particle];
                }
            }
            configuration.triangles.reserve(static_cast<std::size_t>(options.rows - 1u) * (options.columns - 1u) * 2u);
            for (std::uint32_t row = 0u; row + 1u < options.rows; ++row) {
                for (std::uint32_t column = 0u; column + 1u < options.columns; ++column) {
                    const std::uint32_t top_left = row * options.columns + column;
                    const std::uint32_t top_right = top_left + 1u;
                    const std::uint32_t bottom_left = top_left + options.columns;
                    const std::uint32_t bottom_right = bottom_left + 1u;
                    if ((row + column) % 2u == 0u) {
                        configuration.triangles.push_back({.first = top_left, .second = top_right, .third = bottom_right});
                        configuration.triangles.push_back({.first = top_left, .second = bottom_right, .third = bottom_left});
                    } else {
                        configuration.triangles.push_back({.first = top_left, .second = top_right, .third = bottom_left});
                        configuration.triangles.push_back({.first = top_right, .second = bottom_right, .third = bottom_left});
                    }
                }
            }
            return configuration;
        }

        simulation_cuda::Field field(VectorField& value) {
            return {.x = value.x.data, .y = value.y.data, .z = value.z.data};
        }

        simulation_cuda::ConstField field(const VectorField& value) {
            return {.x = value.x.data, .y = value.y.data, .z = value.z.data};
        }

    } // namespace

    ForwardSimulation::ForwardSimulation(ForwardSimulationOptions next_options)
        : options(next_options), model(make_configuration(options)), context(model.make_context(ExecutionMode::forward)), current_state(model.make_state(context)), metrics{}, next_state_(model.make_state(context)), control_(model.make_control(context)), parameters_(model.make_parameters(context)), step_cache_(model.make_step_cache(context)), device_metrics_(static_cast<double*>(context.resource.allocate(10u * sizeof(double)))) {
        context.upload(std::vector<float>(model.configuration.rest_positions.size(), options.mass), parameters_.masses);
        context.upload(std::vector<float>(model.topology.stretch_springs.size(), options.stretch_stiffness), parameters_.stretch_stiffnesses);
        context.upload(std::vector<float>(model.topology.stretch_springs.size(), options.stretch_damping), parameters_.stretch_dampings);
        std::vector<float> stretch_rest_lengths(model.topology.stretch_springs.size());
        for (std::size_t spring = 0u; spring < stretch_rest_lengths.size(); ++spring) stretch_rest_lengths[spring] = model.topology.stretch_springs[spring].rest_length;
        context.upload(stretch_rest_lengths, parameters_.stretch_rest_lengths);
        context.upload(std::vector<float>(model.topology.bending_springs.size(), options.bending_stiffness), parameters_.bending_stiffnesses);
        context.upload(std::vector<float>(model.topology.bending_springs.size(), options.bending_damping), parameters_.bending_dampings);
        std::vector<float> bending_rest_lengths(model.topology.bending_springs.size());
        for (std::size_t spring = 0u; spring < bending_rest_lengths.size(); ++spring) bending_rest_lengths[spring] = model.topology.bending_springs[spring].rest_length;
        context.upload(bending_rest_lengths, parameters_.bending_rest_lengths);
        reset();
    }

    ForwardSimulation::~ForwardSimulation() noexcept {
        context.resource.release(device_metrics_);
    }

    void ForwardSimulation::reset() {
        context.upload(model.configuration.rest_positions, current_state.positions);
        context.upload(model.configuration.rest_positions, next_state_.positions);
        context.resource.zero(current_state.velocities.x.data, current_state.velocities.x.size * sizeof(float));
        context.resource.zero(current_state.velocities.y.data, current_state.velocities.y.size * sizeof(float));
        context.resource.zero(current_state.velocities.z.data, current_state.velocities.z.size * sizeof(float));
        context.resource.zero(next_state_.velocities.x.data, next_state_.velocities.x.size * sizeof(float));
        context.resource.zero(next_state_.velocities.y.data, next_state_.velocities.y.size * sizeof(float));
        context.resource.zero(next_state_.velocities.z.data, next_state_.velocities.z.size * sizeof(float));
        context.resource.zero(control_.external_forces.x.data, control_.external_forces.x.size * sizeof(float));
        context.resource.zero(control_.external_forces.y.data, control_.external_forces.y.size * sizeof(float));
        context.resource.zero(control_.external_forces.z.data, control_.external_forces.z.size * sizeof(float));
        context.resource.synchronize();
        metrics = {};
    }

    void ForwardSimulation::step() {
        const auto begin = std::chrono::steady_clock::now();
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        context.resource.zero(device_metrics_, 10u * sizeof(double));
        simulation_cuda::launch_write_control(stream, options.rows, options.columns, metrics.step, options.time_step, options.mass, options.load_ramp_duration, options.load_period, options.load_base_acceleration, options.load_primary_acceleration, options.load_secondary_acceleration, field(control_.external_forces), device_metrics_);
        model.forward_step(current_state, control_, parameters_, next_state_, step_cache_, context);
        simulation_cuda::launch_particle_metrics(stream, options.rows, options.columns, parameters_.masses.data, field(std::as_const(next_state_.positions)), field(std::as_const(next_state_.velocities)), device_metrics_);
        simulation_cuda::launch_strain_metrics(stream, static_cast<std::uint32_t>(model.topology.stretch_springs.size()), context.device_topology.stretch.first.data, context.device_topology.stretch.second.data, parameters_.stretch_rest_lengths.data, field(std::as_const(next_state_.positions)), device_metrics_ + 2u);
        simulation_cuda::launch_strain_metrics(stream, static_cast<std::uint32_t>(model.topology.bending_springs.size()), context.device_topology.bending.first.data, context.device_topology.bending.second.data, parameters_.bending_rest_lengths.data, field(std::as_const(next_state_.positions)), device_metrics_ + 3u);
        std::array<double, 10u> values{};
        context.resource.copy_to_host(values.data(), device_metrics_, values.size() * sizeof(double));
        context.resource.synchronize();
        std::swap(current_state, next_state_);

        const double step_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        const double previous_steps = static_cast<double>(metrics.step);
        ++metrics.step;
        metrics.physical_time = static_cast<double>(metrics.step) * options.time_step;
        metrics.kinetic_energy = values[0];
        metrics.maximum_velocity = values[1];
        metrics.maximum_absolute_stretch_strain = values[2];
        metrics.maximum_absolute_bending_strain = values[3];
        const float inverse_rows = 1.0F / static_cast<float>(options.rows);
        metrics.free_edge_mean_position = {.x = static_cast<float>(values[4]) * inverse_rows, .y = static_cast<float>(values[5]) * inverse_rows, .z = static_cast<float>(values[6]) * inverse_rows};
        metrics.free_edge_mean_displacement = {.x = metrics.free_edge_mean_position.x - options.width, .y = metrics.free_edge_mean_position.y + 0.5F * options.height, .z = metrics.free_edge_mean_position.z};
        metrics.sampled_load_accelerations = {values[7], values[8], values[9]};
        metrics.step_milliseconds = step_milliseconds;
        metrics.average_step_milliseconds = (previous_steps * metrics.average_step_milliseconds + step_milliseconds) / static_cast<double>(metrics.step);
    }

} // namespace xayah::cloth::examples::forward
