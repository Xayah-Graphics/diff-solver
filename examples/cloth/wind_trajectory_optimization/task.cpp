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

        constexpr std::size_t keyframe_count = 6u;
        constexpr std::size_t variable_count = 2u * keyframe_count;

        task_cuda::ConstField kernel_field(const VectorField& field) {
            return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
        }

        task_cuda::Field kernel_field(VectorField& field) {
            return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
        }

        Vector3 interpolate_wind(const std::span<const Vector3> keyframes, const std::size_t control_step, const std::size_t trajectory_steps) {
            const double coordinate = static_cast<double>(control_step) * static_cast<double>(keyframe_count - 1u) / static_cast<double>(trajectory_steps - 1u);
            const std::size_t first = std::min(static_cast<std::size_t>(coordinate), keyframe_count - 2u);
            const std::size_t second = first + 1u;
            const float weight = static_cast<float>(coordinate - static_cast<double>(first));
            return {
                .x = (1.0F - weight) * keyframes[first].x + weight * keyframes[second].x,
                .y = 0.0F,
                .z = (1.0F - weight) * keyframes[first].z + weight * keyframes[second].z,
            };
        }

    } // namespace

    struct WindTrajectoryOptimizationTask::State final {
        State(Configuration configuration, WindTrajectoryOptimizationOptions options);
        ~State() noexcept;

        void upload_parameters();
        void upload_keyframes(std::span<const Vector3> keyframes);
        void write_controls(std::vector<Control>& controls, std::span<const Vector3> keyframes);
        [[nodiscard]] double trajectory_loss(const xayah::solver::Trajectory<::xayah::cloth::State, StepCache>& trajectory);
        void evaluate(xayah::solver::TapeMode tape_mode);
        [[nodiscard]] double loss_at_keyframes(std::span<const Vector3> keyframes, xayah::solver::TapeMode tape_mode);

        WindTrajectoryOptimizationOptions options;
        Model model;
        ExecutionContext context;
        ::xayah::cloth::State initial_state;
        std::vector<Control> target_controls;
        std::vector<Control> estimated_controls;
        std::vector<Control> probe_controls;
        Parameters parameters;
        xayah::solver::Trajectory<::xayah::cloth::State, StepCache> target_trajectory;
        xayah::solver::Trajectory<::xayah::cloth::State, StepCache> estimated_trajectory;
        xayah::solver::TrajectoryAdjoint<StateAdjoint> trajectory_adjoint;
        std::vector<Vector3> target_keyframes;
        std::vector<Vector3> estimated_keyframes;
        std::array<double, variable_count> keyframe_gradients;
        std::array<double, variable_count> first_moments;
        std::array<double, variable_count> second_moments;
        float* device_keyframes;
        double* device_keyframe_gradients;
        double* scalar;
        WindTrajectoryOptimizationMetrics metrics;
        std::size_t adam_step;
    };

    WindTrajectoryOptimizationTask::State::State(Configuration configuration, WindTrajectoryOptimizationOptions next_options)
        : options(next_options), model(std::move(configuration)), context(model.make_context()), initial_state(model.make_state(context)), target_controls{}, estimated_controls{}, probe_controls{}, parameters(model.make_parameters(context)), target_trajectory{}, estimated_trajectory{}, trajectory_adjoint{}, target_keyframes{
              {.x = 0.00F, .y = 0.0F, .z = 0.00F},
              {.x = 0.30F, .y = 0.0F, .z = 0.12F},
              {.x = 0.18F, .y = 0.0F, .z = -0.18F},
              {.x = -0.18F, .y = 0.0F, .z = 0.18F},
              {.x = -0.30F, .y = 0.0F, .z = -0.12F},
              {.x = 0.00F, .y = 0.0F, .z = 0.00F},
          }, estimated_keyframes(keyframe_count), keyframe_gradients{}, first_moments{}, second_moments{}, device_keyframes(static_cast<float*>(context.resource().allocate(variable_count * sizeof(float)))), device_keyframe_gradients(static_cast<double*>(context.resource().allocate(variable_count * sizeof(double)))), scalar(static_cast<double*>(context.resource().allocate(sizeof(double)))), metrics{}, adam_step{} {
        context.upload(model.configuration().rest_positions, initial_state.positions);
        context.upload(std::vector<Vector3>(model.configuration().rest_positions.size()), initial_state.velocities);
        target_controls.reserve(options.trajectory_steps);
        estimated_controls.reserve(options.trajectory_steps);
        probe_controls.reserve(options.trajectory_steps);
        for (std::size_t step = 0; step < options.trajectory_steps; ++step) {
            target_controls.push_back(model.make_control(context));
            estimated_controls.push_back(model.make_control(context));
            probe_controls.push_back(model.make_control(context));
        }
        upload_parameters();
        write_controls(target_controls, target_keyframes);
        target_trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const Control>{target_controls}, parameters, xayah::solver::TapeMode::recompute_step_cache);
        trajectory_adjoint.states.reserve(target_trajectory.states.size());
        for (std::size_t step = 0; step < target_trajectory.states.size(); ++step) trajectory_adjoint.states.push_back(model.make_state_adjoint(context));
    }

    WindTrajectoryOptimizationTask::State::~State() noexcept {
        context.resource().release(scalar);
        context.resource().release(device_keyframe_gradients);
        context.resource().release(device_keyframes);
    }

    void WindTrajectoryOptimizationTask::State::upload_parameters() {
        context.upload(std::vector<float>(model.configuration().rest_positions.size(), options.mass), parameters.masses);
        context.upload(std::vector<float>(model.topology().stretch_springs.size(), options.stretch_stiffness), parameters.stretch_stiffnesses);
        context.upload(std::vector<float>(model.topology().stretch_springs.size(), options.stretch_damping), parameters.stretch_dampings);
        std::vector<float> stretch_rest_lengths(model.topology().stretch_springs.size());
        for (std::size_t spring = 0; spring < stretch_rest_lengths.size(); ++spring) stretch_rest_lengths[spring] = model.topology().stretch_springs[spring].rest_length;
        context.upload(stretch_rest_lengths, parameters.stretch_rest_lengths);
        context.upload(std::vector<float>(model.topology().bending_springs.size(), options.bending_stiffness), parameters.bending_stiffnesses);
        context.upload(std::vector<float>(model.topology().bending_springs.size(), options.bending_damping), parameters.bending_dampings);
        std::vector<float> bending_rest_lengths(model.topology().bending_springs.size());
        for (std::size_t spring = 0; spring < bending_rest_lengths.size(); ++spring) bending_rest_lengths[spring] = model.topology().bending_springs[spring].rest_length;
        context.upload(bending_rest_lengths, parameters.bending_rest_lengths);
    }

    void WindTrajectoryOptimizationTask::State::upload_keyframes(const std::span<const Vector3> keyframes) {
        std::array<float, variable_count> packed{};
        for (std::size_t keyframe = 0; keyframe < keyframe_count; ++keyframe) {
            packed[2u * keyframe] = keyframes[keyframe].x;
            packed[2u * keyframe + 1u] = keyframes[keyframe].z;
        }
        context.resource().copy_from_host(device_keyframes, packed.data(), packed.size() * sizeof(float));
    }

    void WindTrajectoryOptimizationTask::State::write_controls(std::vector<Control>& controls, const std::span<const Vector3> keyframes) {
        upload_keyframes(keyframes);
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource().native_stream());
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration().rest_positions.size());
        for (std::size_t step = 0; step < controls.size(); ++step)
            task_cuda::launch_write_control(stream, particles, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(options.trajectory_steps), context.device_topology().anchor_mask.data(), device_keyframes, kernel_field(controls[step].external_forces));
    }

    double WindTrajectoryOptimizationTask::State::trajectory_loss(const xayah::solver::Trajectory<::xayah::cloth::State, StepCache>& trajectory) {
        Resource& resource = context.resource();
        const cudaStream_t stream = static_cast<cudaStream_t>(resource.native_stream());
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration().rest_positions.size());
        const double normalization = 1.0 / static_cast<double>(options.trajectory_steps * model.configuration().rest_positions.size());
        resource.zero(scalar, sizeof(double));
        for (std::size_t step = 1; step < trajectory.states.size(); ++step)
            task_cuda::launch_position_loss(stream, particles, normalization, kernel_field(trajectory.states[step].positions), kernel_field(static_cast<const VectorField&>(target_trajectory.states[step].positions)), scalar);
        double loss;
        resource.copy_to_host(&loss, scalar, sizeof(double));
        resource.synchronize();
        return loss;
    }

    void WindTrajectoryOptimizationTask::State::evaluate(const xayah::solver::TapeMode tape_mode) {
        Resource& resource = context.resource();
        const cudaStream_t stream = static_cast<cudaStream_t>(resource.native_stream());
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration().rest_positions.size());
        const float normalization = 1.0F / static_cast<float>(options.trajectory_steps * model.configuration().rest_positions.size());
        write_controls(estimated_controls, estimated_keyframes);
        estimated_trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const Control>{estimated_controls}, parameters, tape_mode);

        resource.zero(scalar, sizeof(double));
        for (std::size_t step = 1; step < estimated_trajectory.states.size(); ++step)
            task_cuda::launch_position_loss_seed(stream, particles, normalization, kernel_field(static_cast<const VectorField&>(estimated_trajectory.states[step].positions)), kernel_field(static_cast<const VectorField&>(target_trajectory.states[step].positions)), kernel_field(trajectory_adjoint.states[step].positions), scalar);
        double loss;
        resource.copy_to_host(&loss, scalar, sizeof(double));

        const auto gradients = xayah::solver::vjp(model, context, estimated_trajectory, std::span<const Control>{estimated_controls}, parameters, trajectory_adjoint);
        resource.zero(device_keyframe_gradients, variable_count * sizeof(double));
        for (std::size_t step = 0; step < gradients.controls.size(); ++step)
            task_cuda::launch_accumulate_keyframe_gradient(stream, particles, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(options.trajectory_steps), context.device_topology().anchor_mask.data(), kernel_field(gradients.controls[step].external_forces), device_keyframe_gradients);
        resource.copy_to_host(keyframe_gradients.data(), device_keyframe_gradients, variable_count * sizeof(double));
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
        for (const double gradient : keyframe_gradients) gradient_norm += gradient * gradient;
        metrics.loss = loss;
        metrics.loss_ratio = loss / metrics.initial_loss;
        metrics.keyframe_relative_error = std::sqrt(error_norm / target_norm);
        metrics.gradient_norm = std::sqrt(gradient_norm);
    }

    double WindTrajectoryOptimizationTask::State::loss_at_keyframes(const std::span<const Vector3> keyframes, const xayah::solver::TapeMode tape_mode) {
        write_controls(probe_controls, keyframes);
        const auto trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const Control>{probe_controls}, parameters, tape_mode);
        return trajectory_loss(trajectory);
    }

    WindTrajectoryOptimizationTask::WindTrajectoryOptimizationTask(Configuration configuration, WindTrajectoryOptimizationOptions options) : state_(std::make_unique<State>(std::move(configuration), options)) {
        reset();
    }

    WindTrajectoryOptimizationTask::WindTrajectoryOptimizationTask(WindTrajectoryOptimizationTask&&) noexcept = default;

    WindTrajectoryOptimizationTask& WindTrajectoryOptimizationTask::operator=(WindTrajectoryOptimizationTask&&) noexcept = default;

    WindTrajectoryOptimizationTask::~WindTrajectoryOptimizationTask() noexcept = default;

    void WindTrajectoryOptimizationTask::reset() {
        std::ranges::fill(state_->estimated_keyframes, Vector3{});
        std::ranges::fill(state_->first_moments, 0.0);
        std::ranges::fill(state_->second_moments, 0.0);
        state_->adam_step = 0u;
        state_->metrics = {.iteration = 0u, .loss = 0.0, .initial_loss = 1.0, .loss_ratio = 1.0, .keyframe_relative_error = 1.0, .gradient_norm = 0.0};
        state_->evaluate(xayah::solver::TapeMode::recompute_step_cache);
        state_->metrics.initial_loss = state_->metrics.loss;
        state_->metrics.loss_ratio = 1.0;
    }

    void WindTrajectoryOptimizationTask::optimize_step() {
        ++state_->adam_step;
        for (std::size_t variable = 0; variable < variable_count; ++variable) {
            const double gradient = state_->keyframe_gradients[variable];
            state_->first_moments[variable] = 0.9 * state_->first_moments[variable] + 0.1 * gradient;
            state_->second_moments[variable] = 0.999 * state_->second_moments[variable] + 0.001 * gradient * gradient;
            const double corrected_first = state_->first_moments[variable] / (1.0 - std::pow(0.9, static_cast<double>(state_->adam_step)));
            const double corrected_second = state_->second_moments[variable] / (1.0 - std::pow(0.999, static_cast<double>(state_->adam_step)));
            const double update = static_cast<double>(state_->options.adam_learning_rate) * corrected_first / (std::sqrt(corrected_second) + 1.0e-8);
            if (variable % 2u == 0u) state_->estimated_keyframes[variable / 2u].x -= static_cast<float>(update);
            else state_->estimated_keyframes[variable / 2u].z -= static_cast<float>(update);
        }
        ++state_->metrics.iteration;
        state_->evaluate(xayah::solver::TapeMode::recompute_step_cache);
    }

    WindTrajectoryGradientCheck WindTrajectoryOptimizationTask::check_gradient(const xayah::solver::TapeMode tape_mode, const float epsilon) {
        state_->evaluate(tape_mode);
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

        StateTangent initial_state_tangent = state_->model.make_state_tangent(state_->context);
        std::vector<ControlTangent> control_tangents;
        control_tangents.reserve(state_->options.trajectory_steps);
        for (std::size_t step = 0; step < state_->options.trajectory_steps; ++step) control_tangents.push_back(state_->model.make_control_tangent(state_->context));
        state_->upload_keyframes(direction);
        const cudaStream_t stream = static_cast<cudaStream_t>(state_->context.resource().native_stream());
        const std::uint32_t particles = static_cast<std::uint32_t>(state_->model.configuration().rest_positions.size());
        for (std::size_t step = 0; step < control_tangents.size(); ++step)
            task_cuda::launch_write_control(stream, particles, static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(state_->options.trajectory_steps), state_->context.device_topology().anchor_mask.data(), state_->device_keyframes, kernel_field(control_tangents[step].external_forces));
        ParameterTangent parameter_tangent = state_->model.make_parameter_tangent(state_->context);
        const auto trajectory_tangent = xayah::solver::jvp(state_->model, state_->context, state_->estimated_trajectory, std::span<const Control>{state_->estimated_controls}, state_->parameters, initial_state_tangent, std::span<const ControlTangent>{control_tangents}, parameter_tangent);

        Resource& resource = state_->context.resource();
        resource.zero(state_->scalar, sizeof(double));
        for (std::size_t step = 1; step < trajectory_tangent.states.size(); ++step)
            task_cuda::launch_position_tangent_inner_product(stream, particles, kernel_field(static_cast<const VectorField&>(state_->trajectory_adjoint.states[step].positions)), kernel_field(trajectory_tangent.states[step].positions), state_->scalar);
        double jvp_inner_product;
        resource.copy_to_host(&jvp_inner_product, state_->scalar, sizeof(double));
        resource.synchronize();

        double vjp_inner_product = 0.0;
        for (std::size_t keyframe = 0; keyframe < keyframe_count; ++keyframe)
            vjp_inner_product += state_->keyframe_gradients[2u * keyframe] * direction[keyframe].x + state_->keyframe_gradients[2u * keyframe + 1u] * direction[keyframe].z;

        std::array<Vector3, keyframe_count> positive_keyframes{};
        std::array<Vector3, keyframe_count> negative_keyframes{};
        for (std::size_t keyframe = 0; keyframe < keyframe_count; ++keyframe) {
            positive_keyframes[keyframe] = state_->estimated_keyframes[keyframe];
            negative_keyframes[keyframe] = state_->estimated_keyframes[keyframe];
            positive_keyframes[keyframe].x += epsilon * direction[keyframe].x;
            positive_keyframes[keyframe].z += epsilon * direction[keyframe].z;
            negative_keyframes[keyframe].x -= epsilon * direction[keyframe].x;
            negative_keyframes[keyframe].z -= epsilon * direction[keyframe].z;
        }
        const double positive_loss = state_->loss_at_keyframes(positive_keyframes, tape_mode);
        const double negative_loss = state_->loss_at_keyframes(negative_keyframes, tape_mode);
        return {
            .finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon),
            .jvp_inner_product = jvp_inner_product,
            .vjp_inner_product = vjp_inner_product,
        };
    }

    const WindTrajectoryOptimizationOptions& WindTrajectoryOptimizationTask::options() const {
        return state_->options;
    }

    const WindTrajectoryOptimizationMetrics& WindTrajectoryOptimizationTask::metrics() const {
        return state_->metrics;
    }

    const Model& WindTrajectoryOptimizationTask::model() const {
        return state_->model;
    }

    ExecutionContext& WindTrajectoryOptimizationTask::context() {
        return state_->context;
    }

    std::size_t WindTrajectoryOptimizationTask::trajectory_state_count() const {
        return state_->target_trajectory.states.size();
    }

    const ::xayah::cloth::State& WindTrajectoryOptimizationTask::target_state(const std::size_t step) const {
        return state_->target_trajectory.states[step];
    }

    const ::xayah::cloth::State& WindTrajectoryOptimizationTask::estimated_state(const std::size_t step) const {
        return state_->estimated_trajectory.states[step];
    }

    std::span<const Vector3> WindTrajectoryOptimizationTask::target_wind_keyframes() const {
        return state_->target_keyframes;
    }

    std::span<const Vector3> WindTrajectoryOptimizationTask::estimated_wind_keyframes() const {
        return state_->estimated_keyframes;
    }

    Vector3 WindTrajectoryOptimizationTask::target_wind(const std::size_t control_step) const {
        return interpolate_wind(state_->target_keyframes, control_step, state_->options.trajectory_steps);
    }

    Vector3 WindTrajectoryOptimizationTask::estimated_wind(const std::size_t control_step) const {
        return interpolate_wind(state_->estimated_keyframes, control_step, state_->options.trajectory_steps);
    }

} // namespace xayah::cloth::examples::wind_trajectory_optimization
