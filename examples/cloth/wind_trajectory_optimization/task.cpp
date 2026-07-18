module;

#include "task.h"

module xayah.examples.cloth.wind_trajectory_optimization;

import std;
import xayah.cloth.data;
import xayah.cloth.model;
import xayah.cloth.runtime;
import xayah.solver;

namespace xayah::cloth::examples::wind_trajectory_optimization {

    namespace {

        task_cuda::ConstField kernel_field(const VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        task_cuda::Field kernel_field(VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

    } // namespace

    WindTrajectoryOptimizationTask::WindTrajectoryOptimizationTask(Configuration configuration, WindTrajectoryOptimizationOptions next_options)
        : options(next_options), model(std::move(configuration)), context(model.make_context()), target_trajectory{}, estimated_trajectory{}, target_keyframes{
              {.x = 0.00F, .y = 0.0F, .z = 0.00F},
              {.x = 0.30F, .y = 0.0F, .z = 0.12F},
              {.x = 0.18F, .y = 0.0F, .z = -0.18F},
              {.x = -0.18F, .y = 0.0F, .z = 0.18F},
              {.x = -0.30F, .y = 0.0F, .z = -0.12F},
              {.x = 0.00F, .y = 0.0F, .z = 0.00F},
          }, estimated_keyframes(keyframe_count), metrics{}, initial_state_(model.make_state(context)), estimated_controls_{}, probe_controls_{}, parameters_(model.make_parameters(context)), trajectory_adjoint_{}, keyframe_gradients_{}, first_moments_{}, second_moments_{}, device_keyframes_(static_cast<float*>(context.resource.allocate(variable_count * sizeof(float)))), device_keyframe_gradients_(static_cast<double*>(context.resource.allocate(variable_count * sizeof(double)))), scalar_(static_cast<double*>(context.resource.allocate(sizeof(double)))), adam_step_{} {
        context.upload(model.configuration.rest_positions, initial_state_.positions);
        context.upload(std::vector<Vector3>(model.configuration.rest_positions.size()), initial_state_.velocities);
        estimated_controls_.reserve(options.trajectory_steps);
        probe_controls_.reserve(options.trajectory_steps);
        for (std::size_t step = 0; step < options.trajectory_steps; ++step) {
            estimated_controls_.push_back(model.make_control(context));
            probe_controls_.push_back(model.make_control(context));
        }
        upload_parameters();
        {
            std::vector<Control> target_controls;
            target_controls.reserve(options.trajectory_steps);
            for (std::size_t step = 0; step < options.trajectory_steps; ++step) target_controls.push_back(model.make_control(context));
            write_controls(target_controls, target_keyframes);
            target_trajectory = xayah::solver::simulate(model, context, initial_state_, std::span<const Control>{target_controls}, parameters_, xayah::solver::TapeMode::recompute_step_cache);
        }
        trajectory_adjoint_.states.reserve(target_trajectory.states.size());
        for (std::size_t step = 0; step < target_trajectory.states.size(); ++step) trajectory_adjoint_.states.push_back(model.make_state_adjoint(context));
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
        evaluate(xayah::solver::TapeMode::recompute_step_cache);
        metrics.initial_loss = metrics.loss;
        metrics.loss_ratio = 1.0;
    }

    void WindTrajectoryOptimizationTask::optimize_step() {
        ++adam_step_;
        for (std::size_t variable = 0; variable < variable_count; ++variable) {
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
        evaluate(xayah::solver::TapeMode::recompute_step_cache);
    }

    WindTrajectoryGradientCheck WindTrajectoryOptimizationTask::check_gradient(const xayah::solver::TapeMode tape_mode, const float epsilon) {
        evaluate(tape_mode);
        std::array<Vector3, keyframe_count> direction{};
        double direction_norm = 0.0;
        for (std::size_t keyframe = 0; keyframe < keyframe_count; ++keyframe) {
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
        for (std::size_t step = 0; step < options.trajectory_steps; ++step) control_tangents.push_back(model.make_control_tangent(context));
        upload_keyframes(direction);
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration.rest_positions.size());
        for (std::size_t step = 0; step < control_tangents.size(); ++step)
            task_cuda::launch_write_control(stream, particles, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(options.trajectory_steps), context.device_topology.anchor_mask.data, device_keyframes_, kernel_field(control_tangents[step].external_forces));
        ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        const auto trajectory_tangent = xayah::solver::jvp(model, context, estimated_trajectory, std::span<const Control>{estimated_controls_}, parameters_, initial_state_tangent, std::span<const ControlTangent>{control_tangents}, parameter_tangent);

        Resource& resource = context.resource;
        resource.zero(scalar_, sizeof(double));
        for (std::size_t step = 1; step < trajectory_tangent.states.size(); ++step)
            task_cuda::launch_position_tangent_inner_product(stream, particles, kernel_field(static_cast<const VectorField&>(trajectory_adjoint_.states[step].positions)), kernel_field(trajectory_tangent.states[step].positions), scalar_);
        double jvp_inner_product;
        resource.copy_to_host(&jvp_inner_product, scalar_, sizeof(double));
        resource.synchronize();

        double vjp_inner_product = 0.0;
        for (std::size_t keyframe = 0; keyframe < keyframe_count; ++keyframe)
            vjp_inner_product += keyframe_gradients_[2u * keyframe] * direction[keyframe].x + keyframe_gradients_[2u * keyframe + 1u] * direction[keyframe].z;

        std::array<Vector3, keyframe_count> positive_keyframes{};
        std::array<Vector3, keyframe_count> negative_keyframes{};
        for (std::size_t keyframe = 0; keyframe < keyframe_count; ++keyframe) {
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
        context.upload(std::vector<float>(model.configuration.rest_positions.size(), options.mass), parameters_.masses);
        context.upload(std::vector<float>(model.topology.stretch_springs.size(), options.stretch_stiffness), parameters_.stretch_stiffnesses);
        context.upload(std::vector<float>(model.topology.stretch_springs.size(), options.stretch_damping), parameters_.stretch_dampings);
        std::vector<float> stretch_rest_lengths(model.topology.stretch_springs.size());
        for (std::size_t spring = 0; spring < stretch_rest_lengths.size(); ++spring) stretch_rest_lengths[spring] = model.topology.stretch_springs[spring].rest_length;
        context.upload(stretch_rest_lengths, parameters_.stretch_rest_lengths);
        context.upload(std::vector<float>(model.topology.bending_springs.size(), options.bending_stiffness), parameters_.bending_stiffnesses);
        context.upload(std::vector<float>(model.topology.bending_springs.size(), options.bending_damping), parameters_.bending_dampings);
        std::vector<float> bending_rest_lengths(model.topology.bending_springs.size());
        for (std::size_t spring = 0; spring < bending_rest_lengths.size(); ++spring) bending_rest_lengths[spring] = model.topology.bending_springs[spring].rest_length;
        context.upload(bending_rest_lengths, parameters_.bending_rest_lengths);
    }

    void WindTrajectoryOptimizationTask::upload_keyframes(const std::span<const Vector3> keyframes) {
        std::array<float, variable_count> packed{};
        for (std::size_t keyframe = 0; keyframe < keyframe_count; ++keyframe) {
            packed[2u * keyframe] = keyframes[keyframe].x;
            packed[2u * keyframe + 1u] = keyframes[keyframe].z;
        }
        context.resource.copy_from_host(device_keyframes_, packed.data(), packed.size() * sizeof(float));
    }

    void WindTrajectoryOptimizationTask::write_controls(std::vector<Control>& controls, const std::span<const Vector3> keyframes) {
        upload_keyframes(keyframes);
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration.rest_positions.size());
        for (std::size_t step = 0; step < controls.size(); ++step)
            task_cuda::launch_write_control(stream, particles, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(options.trajectory_steps), context.device_topology.anchor_mask.data, device_keyframes_, kernel_field(controls[step].external_forces));
    }

    double WindTrajectoryOptimizationTask::trajectory_loss(const xayah::solver::Trajectory<::xayah::cloth::State, StepCache>& trajectory) {
        Resource& resource = context.resource;
        const cudaStream_t stream = static_cast<cudaStream_t>(resource.native_stream);
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration.rest_positions.size());
        const double normalization = 1.0 / static_cast<double>(options.trajectory_steps * model.configuration.rest_positions.size());
        resource.zero(scalar_, sizeof(double));
        for (std::size_t step = 1; step < trajectory.states.size(); ++step)
            task_cuda::launch_position_loss(stream, particles, normalization, kernel_field(trajectory.states[step].positions), kernel_field(static_cast<const VectorField&>(target_trajectory.states[step].positions)), scalar_);
        double loss;
        resource.copy_to_host(&loss, scalar_, sizeof(double));
        resource.synchronize();
        return loss;
    }

    void WindTrajectoryOptimizationTask::evaluate(const xayah::solver::TapeMode tape_mode) {
        Resource& resource = context.resource;
        const cudaStream_t stream = static_cast<cudaStream_t>(resource.native_stream);
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration.rest_positions.size());
        const float normalization = 1.0F / static_cast<float>(options.trajectory_steps * model.configuration.rest_positions.size());
        write_controls(estimated_controls_, estimated_keyframes);
        estimated_trajectory = xayah::solver::simulate(model, context, initial_state_, std::span<const Control>{estimated_controls_}, parameters_, tape_mode);

        resource.zero(scalar_, sizeof(double));
        for (std::size_t step = 1; step < estimated_trajectory.states.size(); ++step)
            task_cuda::launch_position_loss_seed(stream, particles, normalization, kernel_field(static_cast<const VectorField&>(estimated_trajectory.states[step].positions)), kernel_field(static_cast<const VectorField&>(target_trajectory.states[step].positions)), kernel_field(trajectory_adjoint_.states[step].positions), scalar_);
        double loss;
        resource.copy_to_host(&loss, scalar_, sizeof(double));

        const auto gradients = xayah::solver::vjp(model, context, estimated_trajectory, std::span<const Control>{estimated_controls_}, parameters_, trajectory_adjoint_);
        resource.zero(device_keyframe_gradients_, variable_count * sizeof(double));
        for (std::size_t step = 0; step < gradients.controls.size(); ++step)
            task_cuda::launch_accumulate_keyframe_gradient(stream, particles, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(options.trajectory_steps), context.device_topology.anchor_mask.data, kernel_field(gradients.controls[step].external_forces), device_keyframe_gradients_);
        resource.copy_to_host(keyframe_gradients_.data(), device_keyframe_gradients_, variable_count * sizeof(double));
        resource.synchronize();

        double target_norm = 0.0;
        double error_norm = 0.0;
        double gradient_norm = 0.0;
        for (std::size_t keyframe = 0; keyframe < keyframe_count; ++keyframe) {
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

    double WindTrajectoryOptimizationTask::loss_at_keyframes(const std::span<const Vector3> keyframes, const xayah::solver::TapeMode tape_mode) {
        write_controls(probe_controls_, keyframes);
        const auto trajectory = xayah::solver::simulate(model, context, initial_state_, std::span<const Control>{probe_controls_}, parameters_, tape_mode);
        return trajectory_loss(trajectory);
    }

} // namespace xayah::cloth::examples::wind_trajectory_optimization
