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
                .triangles      = {},
                .anchors        = std::vector<std::optional<Vector3>>(static_cast<std::size_t>(options.rows) * options.columns),
                .gravity        = {.x = 0.0F, .y = options.gravity_y, .z = 0.0F},
                .time_step      = options.time_step / static_cast<float>(options.integration_substeps),
            };
            const float spacing_x = options.width / static_cast<float>(options.columns - 1u);
            const float spacing_y = options.height / static_cast<float>(options.rows - 1u);
            for (std::uint32_t row = 0u; row < options.rows; ++row) {
                for (std::uint32_t column = 0u; column < options.columns; ++column) {
                    const std::uint32_t particle           = row * options.columns + column;
                    configuration.rest_positions[particle] = {.x = static_cast<float>(column) * spacing_x, .y = -static_cast<float>(row) * spacing_y, .z = 0.0F};
                    if (column == 0u) configuration.anchors[particle] = configuration.rest_positions[particle];
                }
            }
            configuration.triangles.reserve(static_cast<std::size_t>(options.rows - 1u) * (options.columns - 1u) * 2u);
            for (std::uint32_t row = 0u; row + 1u < options.rows; ++row) {
                for (std::uint32_t column = 0u; column + 1u < options.columns; ++column) {
                    const std::uint32_t top_left     = row * options.columns + column;
                    const std::uint32_t top_right    = top_left + 1u;
                    const std::uint32_t bottom_left  = top_left + options.columns;
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

        cuda_kernel::Field field(VectorField& value) {
            return {.x = value.x.data, .y = value.y.data, .z = value.z.data};
        }

        cuda_kernel::ConstField field(const VectorField& value) {
            return {.x = value.x.data, .y = value.y.data, .z = value.z.data};
        }

    } // namespace

    ForwardSimulation::ForwardSimulation(ForwardSimulationOptions next_options) : options(next_options), model(make_configuration(options)), context(model.allocate_context(ExecutionMode::forward)), current_state(model.allocate_state(context)), parameters(model.allocate_parameters(context)), metrics{}, next_state_(model.allocate_state(context)), control_(model.allocate_control(context)), step_cache_(model.allocate_step_cache(context)), device_metrics_(static_cast<double*>(context.resource->allocate(14u * sizeof(double)))) {
        upload(*context.resource, std::vector<float>(model.configuration.rest_positions.size(), options.mass), parameters.masses);
        upload(*context.resource, std::vector<float>(model.topology.stretch_springs.size(), options.stretch_stiffness), parameters.stretch_stiffnesses);
        upload(*context.resource, std::vector<float>(model.topology.stretch_springs.size(), options.stretch_damping), parameters.stretch_dampings);
        std::vector<float> stretch_rest_lengths(model.topology.stretch_springs.size());
        for (std::size_t spring = 0u; spring < stretch_rest_lengths.size(); ++spring) stretch_rest_lengths[spring] = model.topology.stretch_springs[spring].rest_length;
        upload(*context.resource, stretch_rest_lengths, parameters.stretch_rest_lengths);
        upload(*context.resource, std::vector<float>(model.topology.bending_springs.size(), options.bending_stiffness), parameters.bending_stiffnesses);
        upload(*context.resource, std::vector<float>(model.topology.bending_springs.size(), options.bending_damping), parameters.bending_dampings);
        std::vector<float> bending_rest_lengths(model.topology.bending_springs.size());
        for (std::size_t spring = 0u; spring < bending_rest_lengths.size(); ++spring) bending_rest_lengths[spring] = model.topology.bending_springs[spring].rest_length;
        upload(*context.resource, bending_rest_lengths, parameters.bending_rest_lengths);
        reset();
    }

    ForwardSimulation::~ForwardSimulation() noexcept {
        context.resource->release(device_metrics_);
    }

    void ForwardSimulation::reset() {
        upload(*context.resource, model.configuration.rest_positions, current_state.positions);
        upload(*context.resource, model.configuration.rest_positions, next_state_.positions);
        context.resource->zero(current_state.velocities.x.data, current_state.velocities.x.size * sizeof(float));
        context.resource->zero(current_state.velocities.y.data, current_state.velocities.y.size * sizeof(float));
        context.resource->zero(current_state.velocities.z.data, current_state.velocities.z.size * sizeof(float));
        context.resource->zero(next_state_.velocities.x.data, next_state_.velocities.x.size * sizeof(float));
        context.resource->zero(next_state_.velocities.y.data, next_state_.velocities.y.size * sizeof(float));
        context.resource->zero(next_state_.velocities.z.data, next_state_.velocities.z.size * sizeof(float));
        context.resource->zero(control_.external_forces.x.data, control_.external_forces.x.size * sizeof(float));
        context.resource->zero(control_.external_forces.y.data, control_.external_forces.y.size * sizeof(float));
        context.resource->zero(control_.external_forces.z.data, control_.external_forces.z.size * sizeof(float));
        context.resource->synchronize();
        metrics = {};
    }

    void ForwardSimulation::step() {
        const auto begin          = std::chrono::steady_clock::now();
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource->native_stream);
        const float substep_time_step = options.time_step / static_cast<float>(options.integration_substeps);
        for (std::uint32_t substep = 0u; substep != options.integration_substeps; ++substep) {
            context.resource->zero(device_metrics_, 14u * sizeof(double));
            simulation_cuda::launch_write_control(stream, options.rows, options.columns, metrics.step * options.integration_substeps + substep, substep_time_step, options.width, options.height, options.wind_speed, options.gust_strength, options.gust_frequency, options.air_density, options.drag_coefficient, options.skin_drag_coefficient, options.wind_ramp_duration, field(std::as_const(current_state.positions)), field(std::as_const(current_state.velocities)), field(control_.external_forces), device_metrics_);
            model.forward_step(current_state, control_, parameters, next_state_, step_cache_, context);
            std::swap(current_state, next_state_);
        }
        simulation_cuda::launch_particle_metrics(stream, options.rows, options.columns, parameters.masses.data, field(std::as_const(current_state.positions)), field(std::as_const(current_state.velocities)), device_metrics_);
        simulation_cuda::launch_strain_metrics(stream, static_cast<std::uint32_t>(model.topology.stretch_springs.size()), context.device_topology.stretch.first.data, context.device_topology.stretch.second.data, parameters.stretch_rest_lengths.data, field(std::as_const(current_state.positions)), device_metrics_ + 2u);
        simulation_cuda::launch_strain_metrics(stream, static_cast<std::uint32_t>(model.topology.bending_springs.size()), context.device_topology.bending.first.data, context.device_topology.bending.second.data, parameters.bending_rest_lengths.data, field(std::as_const(current_state.positions)), device_metrics_ + 3u);
        std::array<double, 14u> values{};
        context.resource->copy_to_host(values.data(), device_metrics_, values.size() * sizeof(double));
        context.resource->synchronize();
        const double step_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        const double previous_steps    = static_cast<double>(metrics.step);
        ++metrics.step;
        metrics.physical_time                   = static_cast<double>(metrics.step) * options.time_step;
        metrics.kinetic_energy                  = values[0];
        metrics.maximum_velocity                = values[1];
        metrics.maximum_absolute_stretch_strain = values[2];
        metrics.maximum_absolute_bending_strain = values[3];
        const float inverse_rows                = 1.0F / static_cast<float>(options.rows);
        metrics.free_edge_mean_position         = {.x = static_cast<float>(values[4]) * inverse_rows, .y = static_cast<float>(values[5]) * inverse_rows, .z = static_cast<float>(values[6]) * inverse_rows};
        metrics.free_edge_mean_displacement     = {.x = metrics.free_edge_mean_position.x - options.width, .y = metrics.free_edge_mean_position.y + 0.5F * options.height, .z = metrics.free_edge_mean_position.z};
        metrics.sampled_wind_velocities         = {Vector3{static_cast<float>(values[7]), 0.0F, static_cast<float>(values[8])}, Vector3{static_cast<float>(values[9]), 0.0F, static_cast<float>(values[10])}, Vector3{static_cast<float>(values[11]), 0.0F, static_cast<float>(values[12])}};
        metrics.aerodynamic_force               = values[13];
        metrics.step_milliseconds               = step_milliseconds;
        metrics.average_step_milliseconds       = (previous_steps * metrics.average_step_milliseconds + step_milliseconds) / static_cast<double>(metrics.step);
    }

} // namespace xayah::cloth::examples::forward
