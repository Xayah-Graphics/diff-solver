module;

#include "simulation.h"
#include <cuda_runtime_api.h>

module xayah.examples.smoke.forward;

import std;
import xayah.smoke.data;
import xayah.smoke.model;

namespace xayah::smoke::examples::forward {

    namespace {

        Configuration make_configuration(const ForwardSimulationOptions& options) {
            Configuration configuration{
                .resolution = options.resolution,
                .cell_size = options.cell_size,
                .time_step = options.time_step,
                .pressure_iterations = options.pressure_iterations,
            };
            configuration.velocity_boundary.x_min.mode = VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
            configuration.velocity_boundary.x_max.mode = VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
            configuration.velocity_boundary.y_min.mode = VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
            configuration.velocity_boundary.y_max.mode = VelocityBoundaryMode::zero_gradient;
            configuration.velocity_boundary.z_min.mode = VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
            configuration.velocity_boundary.z_max.mode = VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
            configuration.pressure_boundary.y_max.mode = ScalarBoundaryMode::fixed_value;
            configuration.density_boundary.y_max.mode = ScalarBoundaryMode::fixed_value;
            configuration.temperature_boundary.y_max.mode = ScalarBoundaryMode::fixed_value;
            configuration.vorticity_confinement_enabled = true;
            return configuration;
        }

        simulation_cuda::Grid grid(const Configuration& configuration) {
            return {
                .nx = configuration.resolution[0],
                .ny = configuration.resolution[1],
                .nz = configuration.resolution[2],
                .cell_size = configuration.cell_size,
                .time_step = configuration.time_step,
            };
        }

        simulation_cuda::Vector vector(const Vector3 value) {
            return {.x = value.x, .y = value.y, .z = value.z};
        }

        simulation_cuda::VectorField vector(CenteredVectorField& field) {
            return {.x = field.x.values.data, .y = field.y.values.data, .z = field.z.values.data};
        }

        simulation_cuda::ConstStaggeredVectorField vector(const StaggeredVectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

    } // namespace

    ForwardSimulation::ForwardSimulation(ForwardSimulationOptions next_options)
        : options(next_options), model(make_configuration(options)), context(model.make_context(ExecutionMode::forward)), current_state(model.make_state(context)), metrics{}, next_state_(model.make_state(context)), control_(model.make_control(context)), parameters_(model.make_parameters(context)), step_cache_(model.make_step_cache(context)), device_metrics_(static_cast<double*>(context.resource.allocate(6u * sizeof(double)))) {
        context.upload(options.ambient_temperature, parameters_.ambient_temperature);
        context.upload(options.density_buoyancy, parameters_.density_buoyancy);
        context.upload(options.temperature_buoyancy, parameters_.temperature_buoyancy);
        context.upload(options.vorticity_confinement, parameters_.vorticity_confinement);
        reset();
    }

    ForwardSimulation::~ForwardSimulation() noexcept {
        context.resource.release(device_metrics_);
    }

    void ForwardSimulation::reset() {
        context.resource.zero(current_state.density.values.data, current_state.density.values.size * sizeof(float));
        context.resource.zero(current_state.temperature.values.data, current_state.temperature.values.size * sizeof(float));
        context.resource.zero(current_state.velocity.x.data, current_state.velocity.x.size * sizeof(float));
        context.resource.zero(current_state.velocity.y.data, current_state.velocity.y.size * sizeof(float));
        context.resource.zero(current_state.velocity.z.data, current_state.velocity.z.size * sizeof(float));
        context.resource.zero(next_state_.density.values.data, next_state_.density.values.size * sizeof(float));
        context.resource.zero(next_state_.temperature.values.data, next_state_.temperature.values.size * sizeof(float));
        context.resource.zero(next_state_.velocity.x.data, next_state_.velocity.x.size * sizeof(float));
        context.resource.zero(next_state_.velocity.y.data, next_state_.velocity.y.size * sizeof(float));
        context.resource.zero(next_state_.velocity.z.data, next_state_.velocity.z.size * sizeof(float));
        context.resource.synchronize();
        metrics = {};
    }

    void ForwardSimulation::step() {
        const auto begin = std::chrono::steady_clock::now();
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        simulation_cuda::launch_write_control(stream, grid(model.configuration), metrics.step, options.pulse_period, vector(options.left_source_center), vector(options.right_source_center), options.source_radius, options.density_source_rate, options.temperature_source_rate, vector(options.left_acceleration), vector(options.right_acceleration), control_.density_source.values.data, control_.temperature_source.values.data, vector(control_.external_acceleration));
        model.forward_step(current_state, control_, parameters_, next_state_, step_cache_, context);
        context.resource.zero(device_metrics_, 6u * sizeof(double));
        simulation_cuda::launch_reduce_metrics(stream, grid(model.configuration), context.domain.cell_mask.data, next_state_.density.values.data, next_state_.temperature.values.data, vector(step_cache_.advected_velocity), vector(next_state_.velocity), device_metrics_);
        std::array<double, 6u> values{};
        context.resource.copy_to_host(values.data(), device_metrics_, values.size() * sizeof(double));
        context.resource.synchronize();
        std::swap(current_state, next_state_);

        const double step_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        const double previous_steps = static_cast<double>(metrics.step);
        ++metrics.step;
        metrics.physical_time = static_cast<double>(metrics.step) * options.time_step;
        metrics.density_mass = values[0] * static_cast<double>(options.cell_size) * options.cell_size * options.cell_size;
        metrics.density_maximum = values[1];
        metrics.temperature_maximum = values[2];
        metrics.maximum_velocity = values[3];
        metrics.cfl = values[3] * options.time_step / options.cell_size;
        const double cell_count = static_cast<double>(options.resolution[0]) * options.resolution[1] * options.resolution[2];
        metrics.pre_projection_divergence_rms = std::sqrt(values[4] / cell_count);
        metrics.post_projection_divergence_rms = std::sqrt(values[5] / cell_count);
        metrics.divergence_ratio = metrics.post_projection_divergence_rms / metrics.pre_projection_divergence_rms;
        metrics.step_milliseconds = step_milliseconds;
        metrics.average_step_milliseconds = (previous_steps * metrics.average_step_milliseconds + step_milliseconds) / static_cast<double>(metrics.step);
    }

} // namespace xayah::smoke::examples::forward
