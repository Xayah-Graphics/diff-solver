module;

#include "task.h"

module xayah.examples.smoke.wind_trajectory_optimization;

import std;
import xayah.cuda;
import xayah.smoke.data;
import xayah.smoke.model;
import xayah.solver;

namespace xayah::smoke::examples::wind_trajectory_optimization {

    namespace {

        Configuration make_configuration() {
            Configuration configuration{};
            configuration.velocity_boundary.x_min.mode = VelocityBoundaryMode::periodic;
            configuration.velocity_boundary.x_max.mode = VelocityBoundaryMode::periodic;
            configuration.velocity_boundary.y_min.mode = VelocityBoundaryMode::fixed_value;
            configuration.velocity_boundary.y_max.mode = VelocityBoundaryMode::zero_gradient;
            configuration.velocity_boundary.z_min.mode = VelocityBoundaryMode::periodic;
            configuration.velocity_boundary.z_max.mode = VelocityBoundaryMode::periodic;
            configuration.pressure_boundary.x_min.mode = ScalarBoundaryMode::periodic;
            configuration.pressure_boundary.x_max.mode = ScalarBoundaryMode::periodic;
            configuration.pressure_boundary.y_min.mode = ScalarBoundaryMode::zero_gradient;
            configuration.pressure_boundary.y_max.mode = ScalarBoundaryMode::fixed_value;
            configuration.pressure_boundary.z_min.mode = ScalarBoundaryMode::periodic;
            configuration.pressure_boundary.z_max.mode = ScalarBoundaryMode::periodic;
            configuration.density_boundary.x_min.mode = ScalarBoundaryMode::periodic;
            configuration.density_boundary.x_max.mode = ScalarBoundaryMode::periodic;
            configuration.density_boundary.y_min.mode = ScalarBoundaryMode::zero_gradient;
            configuration.density_boundary.y_max.mode = ScalarBoundaryMode::fixed_value;
            configuration.density_boundary.z_min.mode = ScalarBoundaryMode::periodic;
            configuration.density_boundary.z_max.mode = ScalarBoundaryMode::periodic;
            configuration.temperature_boundary = configuration.density_boundary;
            configuration.vorticity_confinement_enabled = false;
            return configuration;
        }

        task_cuda::Grid task_grid(const Configuration& configuration) {
            return {
                .nx = configuration.resolution[0],
                .ny = configuration.resolution[1],
                .nz = configuration.resolution[2],
                .cell_size = configuration.cell_size,
            };
        }

        task_cuda::VectorField task_vector(CenteredVectorField& field) {
            return {.x = field.x.values.data, .y = field.y.values.data, .z = field.z.values.data};
        }

        task_cuda::ConstAdjointVectorField task_vector(const CenteredVectorAdjointField& field) {
            return {.x = field.x.values.data, .y = field.y.values.data, .z = field.z.values.data};
        }

    } // namespace

    WindTrajectoryOptimizationTask::WindTrajectoryOptimizationTask(WindTrajectoryOptimizationOptions next_options)
        : options(next_options), model(make_configuration()), context(model.make_context(ExecutionMode::differentiable)), target_trajectory{}, estimated_trajectory{}, target_keyframes{
              {.x = 0.0F, .y = 0.0F, .z = 0.0F},
              {.x = 2.0F, .y = 0.0F, .z = 0.8F},
              {.x = 1.2F, .y = 0.0F, .z = -1.2F},
              {.x = -1.2F, .y = 0.0F, .z = 1.2F},
              {.x = -2.0F, .y = 0.0F, .z = -0.8F},
              {.x = 0.0F, .y = 0.0F, .z = 0.0F},
          }, estimated_keyframes(keyframe_count), metrics{}, initial_state_(model.make_state(context)), estimated_controls_{}, probe_controls_{}, parameters_(model.make_parameters(context)), trajectory_adjoint_{}, keyframe_gradients_{}, first_moments_{}, second_moments_{}, device_keyframes_(static_cast<float*>(context.resource.allocate(variable_count * sizeof(float)))), device_keyframe_gradients_(static_cast<double*>(context.resource.allocate(variable_count * sizeof(double)))), scalar_(static_cast<double*>(context.resource.allocate(sizeof(double)))), inverse_target_frame_energies_(options.trajectory_steps), adam_step_{} {
        estimated_controls_.reserve(options.trajectory_steps);
        probe_controls_.reserve(options.trajectory_steps);
        for (std::size_t step = 0u; step < options.trajectory_steps; ++step) {
            estimated_controls_.push_back(model.make_control(context));
            probe_controls_.push_back(model.make_control(context));
        }
        upload_parameters();
        {
            std::vector<Control> target_controls;
            target_controls.reserve(options.trajectory_steps);
            for (std::size_t step = 0u; step < options.trajectory_steps; ++step) target_controls.push_back(model.make_control(context));
            write_controls(target_controls, target_keyframes);
            target_trajectory = solver::simulate(model, context, initial_state_, std::span<const Control>{target_controls}, parameters_, solver::TapeMode::recompute_step_cache);
        }
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const std::uint32_t cell_count = model.configuration.resolution[0] * model.configuration.resolution[1] * model.configuration.resolution[2];
        for (std::size_t step = 1u; step < target_trajectory.states.size(); ++step) {
            context.resource.zero(scalar_, sizeof(double));
            task_cuda::launch_density_energy(stream, cell_count, target_trajectory.states[step].density.values.data, scalar_);
            context.resource.copy_to_host(inverse_target_frame_energies_.data() + step - 1u, scalar_, sizeof(double));
        }
        context.resource.synchronize();
        for (double& energy : inverse_target_frame_energies_) energy = 1.0 / (static_cast<double>(options.trajectory_steps) * energy);
        trajectory_adjoint_.states.reserve(target_trajectory.states.size());
        for (std::size_t step = 0u; step < target_trajectory.states.size(); ++step) trajectory_adjoint_.states.push_back(model.make_state_adjoint(context));
        reset();
    }

    WindTrajectoryOptimizationTask::~WindTrajectoryOptimizationTask() noexcept {
        context.resource.release(scalar_);
        context.resource.release(device_keyframe_gradients_);
        context.resource.release(device_keyframes_);
    }

    void WindTrajectoryOptimizationTask::reset() {
        std::ranges::fill(estimated_keyframes, Vector3{});
        std::ranges::fill(first_moments_, 0.0);
        std::ranges::fill(second_moments_, 0.0);
        adam_step_ = 0u;
        metrics = {.iteration = 0u, .loss = 0.0, .initial_loss = 1.0, .loss_ratio = 1.0, .keyframe_relative_error = 1.0, .gradient_norm = 0.0};
        evaluate(solver::TapeMode::recompute_step_cache);
        metrics.initial_loss = metrics.loss;
        metrics.loss_ratio = 1.0;
    }

    void WindTrajectoryOptimizationTask::optimize_step() {
        ++adam_step_;
        for (std::size_t variable = 0u; variable < variable_count; ++variable) {
            const double gradient = keyframe_gradients_[variable];
            first_moments_[variable] = 0.9 * first_moments_[variable] + 0.1 * gradient;
            second_moments_[variable] = 0.999 * second_moments_[variable] + 0.001 * gradient * gradient;
            const double corrected_first = first_moments_[variable] / (1.0 - std::pow(0.9, static_cast<double>(adam_step_)));
            const double corrected_second = second_moments_[variable] / (1.0 - std::pow(0.999, static_cast<double>(adam_step_)));
            const double update = static_cast<double>(options.adam_learning_rate) * corrected_first / (std::sqrt(corrected_second) + 1.0e-8);
            if (variable % 2u == 0u) estimated_keyframes[variable / 2u].x -= static_cast<float>(update);
            else estimated_keyframes[variable / 2u].z -= static_cast<float>(update);
        }
        ++metrics.iteration;
        evaluate(solver::TapeMode::recompute_step_cache);
    }

    WindTrajectoryGradientCheck WindTrajectoryOptimizationTask::check_gradient(const solver::TapeMode tape_mode, const float epsilon) {
        evaluate(tape_mode);
        std::array<Vector3, keyframe_count> direction{};
        double direction_norm = 0.0;
        for (std::size_t keyframe = 0u; keyframe < keyframe_count; ++keyframe) {
            direction[keyframe].x = static_cast<float>(std::sin(0.71 * static_cast<double>(2u * keyframe + 1u)));
            direction[keyframe].z = static_cast<float>(std::cos(0.53 * static_cast<double>(2u * keyframe + 2u)));
            direction_norm += static_cast<double>(direction[keyframe].x) * direction[keyframe].x + static_cast<double>(direction[keyframe].z) * direction[keyframe].z;
        }
        direction_norm = std::sqrt(direction_norm);
        for (Vector3& value : direction) {
            value.x = static_cast<float>(value.x / direction_norm);
            value.z = static_cast<float>(value.z / direction_norm);
        }

        StateTangent initial_state_tangent = model.make_state_tangent(context);
        std::vector<ControlTangent> control_tangents;
        control_tangents.reserve(options.trajectory_steps);
        for (std::size_t step = 0u; step < options.trajectory_steps; ++step) control_tangents.push_back(model.make_control_tangent(context));
        write_control_tangents(control_tangents, direction);
        ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        const auto trajectory_tangent = solver::jvp(model, context, estimated_trajectory, std::span<const Control>{estimated_controls_}, parameters_, initial_state_tangent, std::span<const ControlTangent>{control_tangents}, parameter_tangent);

        context.resource.zero(scalar_, sizeof(double));
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const std::uint32_t cell_count = model.configuration.resolution[0] * model.configuration.resolution[1] * model.configuration.resolution[2];
        for (std::size_t step = 1u; step < trajectory_tangent.states.size(); ++step)
            task_cuda::launch_density_tangent_inner_product(stream, cell_count, trajectory_adjoint_.states[step].density.values.data, trajectory_tangent.states[step].density.values.data, scalar_);
        double jvp_inner_product;
        context.resource.copy_to_host(&jvp_inner_product, scalar_, sizeof(double));
        context.resource.synchronize();

        double vjp_inner_product = 0.0;
        for (std::size_t keyframe = 0u; keyframe < keyframe_count; ++keyframe)
            vjp_inner_product += keyframe_gradients_[2u * keyframe] * direction[keyframe].x + keyframe_gradients_[2u * keyframe + 1u] * direction[keyframe].z;

        std::array<Vector3, keyframe_count> positive_keyframes{};
        std::array<Vector3, keyframe_count> negative_keyframes{};
        for (std::size_t keyframe = 0u; keyframe < keyframe_count; ++keyframe) {
            positive_keyframes[keyframe] = estimated_keyframes[keyframe];
            negative_keyframes[keyframe] = estimated_keyframes[keyframe];
            positive_keyframes[keyframe].x += epsilon * direction[keyframe].x;
            positive_keyframes[keyframe].z += epsilon * direction[keyframe].z;
            negative_keyframes[keyframe].x -= epsilon * direction[keyframe].x;
            negative_keyframes[keyframe].z -= epsilon * direction[keyframe].z;
        }
        const double positive_loss = loss_at_keyframes(positive_keyframes, tape_mode);
        const double negative_loss = loss_at_keyframes(negative_keyframes, tape_mode);
        return {
            .finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon),
            .jvp_inner_product = jvp_inner_product,
            .vjp_inner_product = vjp_inner_product,
        };
    }

    void WindTrajectoryOptimizationTask::upload_parameters() {
        context.upload(options.ambient_temperature, parameters_.ambient_temperature);
        context.upload(options.density_buoyancy, parameters_.density_buoyancy);
        context.upload(options.temperature_buoyancy, parameters_.temperature_buoyancy);
        context.upload(0.0F, parameters_.vorticity_confinement);
    }

    void WindTrajectoryOptimizationTask::upload_keyframes(const std::span<const Vector3> keyframes) {
        std::array<float, variable_count> packed{};
        for (std::size_t keyframe = 0u; keyframe < keyframe_count; ++keyframe) {
            packed[2u * keyframe] = keyframes[keyframe].x;
            packed[2u * keyframe + 1u] = keyframes[keyframe].z;
        }
        context.resource.copy_from_host(device_keyframes_, packed.data(), packed.size() * sizeof(float));
    }

    void WindTrajectoryOptimizationTask::write_controls(std::vector<Control>& controls, const std::span<const Vector3> keyframes) {
        upload_keyframes(keyframes);
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const task_cuda::Grid grid = task_grid(model.configuration);
        for (std::size_t step = 0u; step < controls.size(); ++step)
            task_cuda::launch_write_control(stream, grid, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(options.trajectory_steps), device_keyframes_, options.density_source_rate, options.temperature_source_rate, controls[step].density_source.values.data, controls[step].temperature_source.values.data, task_vector(controls[step].external_acceleration));
    }

    void WindTrajectoryOptimizationTask::write_control_tangents(std::vector<ControlTangent>& controls, const std::span<const Vector3> keyframes) {
        upload_keyframes(keyframes);
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const task_cuda::Grid grid = task_grid(model.configuration);
        for (std::size_t step = 0u; step < controls.size(); ++step)
            task_cuda::launch_write_control_tangent(stream, grid, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(options.trajectory_steps), device_keyframes_, controls[step].density_source.values.data, controls[step].temperature_source.values.data, task_vector(controls[step].external_acceleration));
    }

    double WindTrajectoryOptimizationTask::trajectory_loss(const solver::Trajectory<State, StepCache>& trajectory) {
        context.resource.zero(scalar_, sizeof(double));
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const std::uint32_t cell_count = model.configuration.resolution[0] * model.configuration.resolution[1] * model.configuration.resolution[2];
        for (std::size_t step = 1u; step < trajectory.states.size(); ++step)
            task_cuda::launch_density_loss(stream, cell_count, inverse_target_frame_energies_[step - 1u], trajectory.states[step].density.values.data, target_trajectory.states[step].density.values.data, scalar_);
        double loss;
        context.resource.copy_to_host(&loss, scalar_, sizeof(double));
        context.resource.synchronize();
        return loss;
    }

    void WindTrajectoryOptimizationTask::evaluate(const solver::TapeMode tape_mode) {
        write_controls(estimated_controls_, estimated_keyframes);
        estimated_trajectory = solver::simulate(model, context, initial_state_, std::span<const Control>{estimated_controls_}, parameters_, tape_mode);

        context.resource.zero(scalar_, sizeof(double));
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const std::uint32_t cell_count = model.configuration.resolution[0] * model.configuration.resolution[1] * model.configuration.resolution[2];
        for (std::size_t step = 1u; step < estimated_trajectory.states.size(); ++step)
            task_cuda::launch_density_loss_seed(stream, cell_count, inverse_target_frame_energies_[step - 1u], estimated_trajectory.states[step].density.values.data, target_trajectory.states[step].density.values.data, trajectory_adjoint_.states[step].density.values.data, scalar_);
        double loss;
        context.resource.copy_to_host(&loss, scalar_, sizeof(double));

        const auto gradients = solver::vjp(model, context, estimated_trajectory, std::span<const Control>{estimated_controls_}, parameters_, trajectory_adjoint_);
        context.resource.zero(device_keyframe_gradients_, variable_count * sizeof(double));
        const task_cuda::Grid grid = task_grid(model.configuration);
        for (std::size_t step = 0u; step < gradients.controls.size(); ++step)
            task_cuda::launch_accumulate_keyframe_gradient(stream, grid, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(options.trajectory_steps), task_vector(static_cast<const CenteredVectorAdjointField&>(gradients.controls[step].external_acceleration)), device_keyframe_gradients_);
        context.resource.copy_to_host(keyframe_gradients_.data(), device_keyframe_gradients_, variable_count * sizeof(double));
        context.resource.synchronize();

        double target_norm = 0.0;
        double error_norm = 0.0;
        double gradient_norm = 0.0;
        for (std::size_t keyframe = 0u; keyframe < keyframe_count; ++keyframe) {
            target_norm += static_cast<double>(target_keyframes[keyframe].x) * target_keyframes[keyframe].x + static_cast<double>(target_keyframes[keyframe].z) * target_keyframes[keyframe].z;
            const double x_error = static_cast<double>(estimated_keyframes[keyframe].x) - target_keyframes[keyframe].x;
            const double z_error = static_cast<double>(estimated_keyframes[keyframe].z) - target_keyframes[keyframe].z;
            error_norm += x_error * x_error + z_error * z_error;
        }
        for (const double gradient : keyframe_gradients_) gradient_norm += gradient * gradient;
        metrics.loss = loss;
        metrics.loss_ratio = loss / metrics.initial_loss;
        metrics.keyframe_relative_error = std::sqrt(error_norm / target_norm);
        metrics.gradient_norm = std::sqrt(gradient_norm);
    }

    double WindTrajectoryOptimizationTask::loss_at_keyframes(const std::span<const Vector3> keyframes, const solver::TapeMode tape_mode) {
        write_controls(probe_controls_, keyframes);
        const auto trajectory = solver::simulate(model, context, initial_state_, std::span<const Control>{probe_controls_}, parameters_, tape_mode);
        return trajectory_loss(trajectory);
    }

} // namespace xayah::smoke::examples::wind_trajectory_optimization
