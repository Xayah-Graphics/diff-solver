module;

#include "task.h"

module xayah.examples.cloth.stretch_stiffness_inverse;

import std;
import xayah.cloth.data;
import xayah.cloth.model;
import xayah.cloth.runtime;
import xayah.solver;

namespace xayah::cloth::examples::stretch_stiffness_inverse {

    namespace {

        inverse_cuda::ConstField kernel_field(const VectorField& field) {
            return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
        }

        inverse_cuda::Field kernel_field(VectorField& field) {
            return {.x = field.x.data(), .y = field.y.data(), .z = field.z.data()};
        }

    } // namespace

    struct StretchStiffnessInverseTask::State final {
        State(Configuration configuration, StretchStiffnessInverseOptions options);
        ~State() noexcept;

        void upload_parameters(Parameters& parameters, float stretch_stiffness);
        [[nodiscard]] double trajectory_loss(const xayah::solver::Trajectory<::xayah::cloth::State, StepCache>& trajectory);
        void evaluate(xayah::solver::TapeMode tape_mode);
        [[nodiscard]] double loss_at_log_stiffness(double log_stiffness, xayah::solver::TapeMode tape_mode);

        StretchStiffnessInverseOptions options;
        Model model;
        ExecutionContext context;
        ::xayah::cloth::State initial_state;
        std::vector<Control> controls;
        Parameters target_parameters;
        Parameters estimated_parameters;
        xayah::solver::Trajectory<::xayah::cloth::State, StepCache> target_trajectory;
        xayah::solver::Trajectory<::xayah::cloth::State, StepCache> estimated_trajectory;
        xayah::solver::TrajectoryAdjoint<StateAdjoint> trajectory_adjoint;
        double* scalar;
        StretchStiffnessInverseMetrics metrics;
        double log_stiffness;
        double first_moment;
        double second_moment;
        std::size_t adam_step;
    };

    StretchStiffnessInverseTask::State::State(Configuration configuration, StretchStiffnessInverseOptions next_options)
        : options(next_options), model(std::move(configuration)), context(model.make_context()), initial_state(model.make_state(context)), controls{}, target_parameters(model.make_parameters(context)), estimated_parameters(model.make_parameters(context)), target_trajectory{}, estimated_trajectory{}, trajectory_adjoint{}, scalar(static_cast<double*>(context.resource().allocate(sizeof(double)))), metrics{}, log_stiffness{}, first_moment{}, second_moment{}, adam_step{} {
        context.upload(model.configuration().rest_positions, initial_state.positions);
        context.upload(std::vector<Vector3>(model.configuration().rest_positions.size()), initial_state.velocities);
        controls.reserve(options.trajectory_steps);
        for (std::size_t step = 0; step < options.trajectory_steps; ++step) controls.push_back(model.make_control(context));
        upload_parameters(target_parameters, options.target_stretch_stiffness);
        upload_parameters(estimated_parameters, options.initial_stretch_stiffness);
        target_trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const Control>{controls}, target_parameters, xayah::solver::TapeMode::recompute_step_cache);
        trajectory_adjoint.states.reserve(target_trajectory.states.size());
        for (std::size_t step = 0; step < target_trajectory.states.size(); ++step) trajectory_adjoint.states.push_back(model.make_state_adjoint(context));
    }

    StretchStiffnessInverseTask::State::~State() noexcept {
        context.resource().release(scalar);
    }

    void StretchStiffnessInverseTask::State::upload_parameters(Parameters& parameters, const float stretch_stiffness) {
        context.upload(std::vector<float>(model.configuration().rest_positions.size(), options.mass), parameters.masses);
        context.upload(std::vector<float>(model.topology().stretch_springs.size(), stretch_stiffness), parameters.stretch_stiffnesses);
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

    double StretchStiffnessInverseTask::State::trajectory_loss(const xayah::solver::Trajectory<::xayah::cloth::State, StepCache>& trajectory) {
        Resource& resource              = context.resource();
        const cudaStream_t stream       = static_cast<cudaStream_t>(resource.native_stream());
        const std::uint32_t particles   = static_cast<std::uint32_t>(model.configuration().rest_positions.size());
        const double normalization      = 1.0 / static_cast<double>(options.trajectory_steps * model.configuration().rest_positions.size());
        resource.zero(scalar, sizeof(double));
        for (std::size_t step = 1; step < trajectory.states.size(); ++step) inverse_cuda::launch_position_loss(stream, particles, normalization, kernel_field(trajectory.states[step].positions), kernel_field(static_cast<const VectorField&>(target_trajectory.states[step].positions)), scalar);
        double loss;
        resource.copy_to_host(&loss, scalar, sizeof(double));
        resource.synchronize();
        return loss;
    }

    void StretchStiffnessInverseTask::State::evaluate(const xayah::solver::TapeMode tape_mode) {
        Resource& resource            = context.resource();
        const cudaStream_t stream     = static_cast<cudaStream_t>(resource.native_stream());
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration().rest_positions.size());
        const float stiffness         = static_cast<float>(std::exp(log_stiffness));
        const float normalization     = 1.0F / static_cast<float>(options.trajectory_steps * model.configuration().rest_positions.size());
        inverse_cuda::launch_fill(stream, static_cast<std::uint32_t>(estimated_parameters.stretch_stiffnesses.size()), stiffness, estimated_parameters.stretch_stiffnesses.data());
        estimated_trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const Control>{controls}, estimated_parameters, tape_mode);

        resource.zero(scalar, sizeof(double));
        for (std::size_t step = 1; step < estimated_trajectory.states.size(); ++step)
            inverse_cuda::launch_position_loss_seed(stream, particles, normalization, kernel_field(static_cast<const VectorField&>(estimated_trajectory.states[step].positions)), kernel_field(static_cast<const VectorField&>(target_trajectory.states[step].positions)), kernel_field(trajectory_adjoint.states[step].positions), scalar);
        double loss;
        resource.copy_to_host(&loss, scalar, sizeof(double));
        resource.synchronize();

        const auto gradients = xayah::solver::vjp(model, context, estimated_trajectory, std::span<const Control>{controls}, estimated_parameters, trajectory_adjoint);
        resource.zero(scalar, sizeof(double));
        inverse_cuda::launch_sum(stream, static_cast<std::uint32_t>(gradients.parameters.stretch_stiffnesses.size()), gradients.parameters.stretch_stiffnesses.data(), scalar);
        double stiffness_gradient;
        resource.copy_to_host(&stiffness_gradient, scalar, sizeof(double));
        resource.synchronize();

        metrics.stretch_stiffness      = stiffness;
        metrics.loss                   = loss;
        metrics.log_stiffness_gradient = static_cast<double>(stiffness) * stiffness_gradient;
    }

    double StretchStiffnessInverseTask::State::loss_at_log_stiffness(const double value, const xayah::solver::TapeMode tape_mode) {
        Parameters parameters = model.make_parameters(context);
        upload_parameters(parameters, static_cast<float>(std::exp(value)));
        const auto trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const Control>{controls}, parameters, tape_mode);
        return trajectory_loss(trajectory);
    }

    StretchStiffnessInverseTask::StretchStiffnessInverseTask(Configuration configuration, StretchStiffnessInverseOptions options) : state_(std::make_unique<State>(std::move(configuration), options)) {
        reset();
    }

    StretchStiffnessInverseTask::StretchStiffnessInverseTask(StretchStiffnessInverseTask&&) noexcept = default;

    StretchStiffnessInverseTask& StretchStiffnessInverseTask::operator=(StretchStiffnessInverseTask&&) noexcept = default;

    StretchStiffnessInverseTask::~StretchStiffnessInverseTask() noexcept = default;

    void StretchStiffnessInverseTask::reset() {
        state_->log_stiffness = std::log(static_cast<double>(state_->options.initial_stretch_stiffness));
        state_->first_moment  = 0.0;
        state_->second_moment = 0.0;
        state_->adam_step     = 0u;
        state_->metrics       = {.iteration = 0u, .stretch_stiffness = state_->options.initial_stretch_stiffness, .loss = 0.0, .initial_loss = 0.0, .log_stiffness_gradient = 0.0};
        state_->evaluate(xayah::solver::TapeMode::recompute_step_cache);
        state_->metrics.initial_loss = state_->metrics.loss;
    }

    void StretchStiffnessInverseTask::optimize_step() {
        ++state_->adam_step;
        const double gradient          = state_->metrics.log_stiffness_gradient;
        state_->first_moment           = 0.9 * state_->first_moment + 0.1 * gradient;
        state_->second_moment          = 0.999 * state_->second_moment + 0.001 * gradient * gradient;
        const double corrected_first   = state_->first_moment / (1.0 - std::pow(0.9, static_cast<double>(state_->adam_step)));
        const double corrected_second  = state_->second_moment / (1.0 - std::pow(0.999, static_cast<double>(state_->adam_step)));
        state_->log_stiffness          -= static_cast<double>(state_->options.adam_learning_rate) * corrected_first / (std::sqrt(corrected_second) + 1.0e-8);
        ++state_->metrics.iteration;
        state_->evaluate(xayah::solver::TapeMode::recompute_step_cache);
    }

    StretchStiffnessGradientCheck StretchStiffnessInverseTask::check_gradient(const xayah::solver::TapeMode tape_mode, const float epsilon) {
        state_->evaluate(tape_mode);
        StateTangent initial_state_tangent = state_->model.make_state_tangent(state_->context);
        std::vector<ControlTangent> control_tangents;
        control_tangents.reserve(state_->controls.size());
        for (std::size_t step = 0; step < state_->controls.size(); ++step) control_tangents.push_back(state_->model.make_control_tangent(state_->context));
        ParameterTangent parameter_tangent = state_->model.make_parameter_tangent(state_->context);
        inverse_cuda::launch_fill(static_cast<cudaStream_t>(state_->context.resource().native_stream()), static_cast<std::uint32_t>(parameter_tangent.stretch_stiffnesses.size()), state_->metrics.stretch_stiffness, parameter_tangent.stretch_stiffnesses.data());
        const auto trajectory_tangent = xayah::solver::jvp(state_->model, state_->context, state_->estimated_trajectory, std::span<const Control>{state_->controls}, state_->estimated_parameters, initial_state_tangent, std::span<const ControlTangent>{control_tangents}, parameter_tangent);

        Resource& resource        = state_->context.resource();
        const cudaStream_t stream = static_cast<cudaStream_t>(resource.native_stream());
        resource.zero(state_->scalar, sizeof(double));
        for (std::size_t step = 1; step < trajectory_tangent.states.size(); ++step)
            inverse_cuda::launch_position_tangent_inner_product(stream, static_cast<std::uint32_t>(state_->model.configuration().rest_positions.size()), kernel_field(static_cast<const VectorField&>(state_->trajectory_adjoint.states[step].positions)), kernel_field(trajectory_tangent.states[step].positions), state_->scalar);
        double jvp_inner_product;
        resource.copy_to_host(&jvp_inner_product, state_->scalar, sizeof(double));
        resource.synchronize();

        const double positive_loss = state_->loss_at_log_stiffness(state_->log_stiffness + epsilon, tape_mode);
        const double negative_loss = state_->loss_at_log_stiffness(state_->log_stiffness - epsilon, tape_mode);
        return {
            .finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon),
            .jvp_inner_product = jvp_inner_product,
            .vjp_inner_product = state_->metrics.log_stiffness_gradient,
        };
    }

    const StretchStiffnessInverseOptions& StretchStiffnessInverseTask::options() const {
        return state_->options;
    }

    const StretchStiffnessInverseMetrics& StretchStiffnessInverseTask::metrics() const {
        return state_->metrics;
    }

    const Model& StretchStiffnessInverseTask::model() const {
        return state_->model;
    }

    ExecutionContext& StretchStiffnessInverseTask::context() {
        return state_->context;
    }

    std::size_t StretchStiffnessInverseTask::trajectory_state_count() const {
        return state_->target_trajectory.states.size();
    }

    const ::xayah::cloth::State& StretchStiffnessInverseTask::target_state(const std::size_t step) const {
        return state_->target_trajectory.states[step];
    }

    const ::xayah::cloth::State& StretchStiffnessInverseTask::estimated_state(const std::size_t step) const {
        return state_->estimated_trajectory.states[step];
    }

} // namespace xayah::cloth::examples::stretch_stiffness_inverse
