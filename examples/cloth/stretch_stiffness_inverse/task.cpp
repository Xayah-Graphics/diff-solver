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
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

        inverse_cuda::Field kernel_field(VectorField& field) {
            return {.x = field.x.data, .y = field.y.data, .z = field.z.data};
        }

    } // namespace

    StretchStiffnessInverseTask::StretchStiffnessInverseTask(Configuration configuration, StretchStiffnessInverseOptions next_options)
        : options(next_options), model(std::move(configuration)), context(model.make_context()), target_trajectory{}, estimated_trajectory{}, metrics{}, initial_state_(model.make_state(context)), controls_{}, estimated_parameters_(model.make_parameters(context)), trajectory_adjoint_{}, scalar_(static_cast<double*>(context.resource.allocate(sizeof(double)))), log_stiffness_{}, first_moment_{}, second_moment_{}, adam_step_{} {
        context.upload(model.configuration.rest_positions, initial_state_.positions);
        context.upload(std::vector<Vector3>(model.configuration.rest_positions.size()), initial_state_.velocities);
        controls_.reserve(options.trajectory_steps);
        for (std::size_t step = 0; step < options.trajectory_steps; ++step) controls_.push_back(model.make_control(context));
        upload_parameters(estimated_parameters_, options.initial_stretch_stiffness);
        {
            Parameters target_parameters = model.make_parameters(context);
            upload_parameters(target_parameters, options.target_stretch_stiffness);
            target_trajectory = xayah::solver::simulate(model, context, initial_state_, std::span<const Control>{controls_}, target_parameters, xayah::solver::TapeMode::recompute_step_cache);
        }
        trajectory_adjoint_.states.reserve(target_trajectory.states.size());
        for (std::size_t step = 0; step < target_trajectory.states.size(); ++step) trajectory_adjoint_.states.push_back(model.make_state_adjoint(context));
        reset();
    }

    StretchStiffnessInverseTask::~StretchStiffnessInverseTask() noexcept {
        context.resource.release(scalar_);
    }

    void StretchStiffnessInverseTask::reset() {
        log_stiffness_ = std::log(static_cast<double>(options.initial_stretch_stiffness));
        first_moment_  = 0.0;
        second_moment_ = 0.0;
        adam_step_     = 0u;
        metrics        = {.iteration = 0u, .stretch_stiffness = options.initial_stretch_stiffness, .loss = 0.0, .initial_loss = 0.0, .log_stiffness_gradient = 0.0};
        evaluate(xayah::solver::TapeMode::recompute_step_cache);
        metrics.initial_loss = metrics.loss;
    }

    void StretchStiffnessInverseTask::optimize_step() {
        ++adam_step_;
        const double gradient          = metrics.log_stiffness_gradient;
        first_moment_                  = 0.9 * first_moment_ + 0.1 * gradient;
        second_moment_                 = 0.999 * second_moment_ + 0.001 * gradient * gradient;
        const double corrected_first   = first_moment_ / (1.0 - std::pow(0.9, static_cast<double>(adam_step_)));
        const double corrected_second  = second_moment_ / (1.0 - std::pow(0.999, static_cast<double>(adam_step_)));
        log_stiffness_                 -= static_cast<double>(options.adam_learning_rate) * corrected_first / (std::sqrt(corrected_second) + 1.0e-8);
        ++metrics.iteration;
        evaluate(xayah::solver::TapeMode::recompute_step_cache);
    }

    StretchStiffnessGradientCheck StretchStiffnessInverseTask::check_gradient(const xayah::solver::TapeMode tape_mode, const float epsilon) {
        evaluate(tape_mode);
        StateTangent initial_state_tangent = model.make_state_tangent(context);
        std::vector<ControlTangent> control_tangents;
        control_tangents.reserve(controls_.size());
        for (std::size_t step = 0; step < controls_.size(); ++step) control_tangents.push_back(model.make_control_tangent(context));
        ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        inverse_cuda::launch_fill(static_cast<cudaStream_t>(context.resource.native_stream), static_cast<std::uint32_t>(parameter_tangent.stretch_stiffnesses.size), metrics.stretch_stiffness, parameter_tangent.stretch_stiffnesses.data);
        const auto trajectory_tangent = xayah::solver::jvp(model, context, estimated_trajectory, std::span<const Control>{controls_}, estimated_parameters_, initial_state_tangent, std::span<const ControlTangent>{control_tangents}, parameter_tangent);

        Resource& resource        = context.resource;
        const cudaStream_t stream = static_cast<cudaStream_t>(resource.native_stream);
        resource.zero(scalar_, sizeof(double));
        for (std::size_t step = 1; step < trajectory_tangent.states.size(); ++step)
            inverse_cuda::launch_position_tangent_inner_product(stream, static_cast<std::uint32_t>(model.configuration.rest_positions.size()), kernel_field(static_cast<const VectorField&>(trajectory_adjoint_.states[step].positions)), kernel_field(trajectory_tangent.states[step].positions), scalar_);
        double jvp_inner_product;
        resource.copy_to_host(&jvp_inner_product, scalar_, sizeof(double));
        resource.synchronize();

        const double positive_loss = loss_at_log_stiffness(log_stiffness_ + epsilon, tape_mode);
        const double negative_loss = loss_at_log_stiffness(log_stiffness_ - epsilon, tape_mode);
        return {
            .finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon),
            .jvp_inner_product = jvp_inner_product,
            .vjp_inner_product = metrics.log_stiffness_gradient,
        };
    }

    void StretchStiffnessInverseTask::upload_parameters(Parameters& parameters, const float stretch_stiffness) {
        context.upload(std::vector<float>(model.configuration.rest_positions.size(), options.mass), parameters.masses);
        context.upload(std::vector<float>(model.topology.stretch_springs.size(), stretch_stiffness), parameters.stretch_stiffnesses);
        context.upload(std::vector<float>(model.topology.stretch_springs.size(), options.stretch_damping), parameters.stretch_dampings);
        std::vector<float> stretch_rest_lengths(model.topology.stretch_springs.size());
        for (std::size_t spring = 0; spring < stretch_rest_lengths.size(); ++spring) stretch_rest_lengths[spring] = model.topology.stretch_springs[spring].rest_length;
        context.upload(stretch_rest_lengths, parameters.stretch_rest_lengths);
        context.upload(std::vector<float>(model.topology.bending_springs.size(), options.bending_stiffness), parameters.bending_stiffnesses);
        context.upload(std::vector<float>(model.topology.bending_springs.size(), options.bending_damping), parameters.bending_dampings);
        std::vector<float> bending_rest_lengths(model.topology.bending_springs.size());
        for (std::size_t spring = 0; spring < bending_rest_lengths.size(); ++spring) bending_rest_lengths[spring] = model.topology.bending_springs[spring].rest_length;
        context.upload(bending_rest_lengths, parameters.bending_rest_lengths);
    }

    double StretchStiffnessInverseTask::trajectory_loss(const xayah::solver::Trajectory<::xayah::cloth::State, StepCache>& trajectory) {
        Resource& resource              = context.resource;
        const cudaStream_t stream       = static_cast<cudaStream_t>(resource.native_stream);
        const std::uint32_t particles   = static_cast<std::uint32_t>(model.configuration.rest_positions.size());
        const double normalization      = 1.0 / static_cast<double>(options.trajectory_steps * model.configuration.rest_positions.size());
        resource.zero(scalar_, sizeof(double));
        for (std::size_t step = 1; step < trajectory.states.size(); ++step) inverse_cuda::launch_position_loss(stream, particles, normalization, kernel_field(trajectory.states[step].positions), kernel_field(static_cast<const VectorField&>(target_trajectory.states[step].positions)), scalar_);
        double loss;
        resource.copy_to_host(&loss, scalar_, sizeof(double));
        resource.synchronize();
        return loss;
    }

    void StretchStiffnessInverseTask::evaluate(const xayah::solver::TapeMode tape_mode) {
        Resource& resource            = context.resource;
        const cudaStream_t stream     = static_cast<cudaStream_t>(resource.native_stream);
        const std::uint32_t particles = static_cast<std::uint32_t>(model.configuration.rest_positions.size());
        const float stiffness         = static_cast<float>(std::exp(log_stiffness_));
        const float normalization     = 1.0F / static_cast<float>(options.trajectory_steps * model.configuration.rest_positions.size());
        inverse_cuda::launch_fill(stream, static_cast<std::uint32_t>(estimated_parameters_.stretch_stiffnesses.size), stiffness, estimated_parameters_.stretch_stiffnesses.data);
        estimated_trajectory = xayah::solver::simulate(model, context, initial_state_, std::span<const Control>{controls_}, estimated_parameters_, tape_mode);

        resource.zero(scalar_, sizeof(double));
        for (std::size_t step = 1; step < estimated_trajectory.states.size(); ++step)
            inverse_cuda::launch_position_loss_seed(stream, particles, normalization, kernel_field(static_cast<const VectorField&>(estimated_trajectory.states[step].positions)), kernel_field(static_cast<const VectorField&>(target_trajectory.states[step].positions)), kernel_field(trajectory_adjoint_.states[step].positions), scalar_);
        double loss;
        resource.copy_to_host(&loss, scalar_, sizeof(double));
        resource.synchronize();

        const auto gradients = xayah::solver::vjp(model, context, estimated_trajectory, std::span<const Control>{controls_}, estimated_parameters_, trajectory_adjoint_);
        resource.zero(scalar_, sizeof(double));
        inverse_cuda::launch_sum(stream, static_cast<std::uint32_t>(gradients.parameters.stretch_stiffnesses.size), gradients.parameters.stretch_stiffnesses.data, scalar_);
        double stiffness_gradient;
        resource.copy_to_host(&stiffness_gradient, scalar_, sizeof(double));
        resource.synchronize();

        metrics.stretch_stiffness      = stiffness;
        metrics.loss                   = loss;
        metrics.log_stiffness_gradient = static_cast<double>(stiffness) * stiffness_gradient;
    }

    double StretchStiffnessInverseTask::loss_at_log_stiffness(const double value, const xayah::solver::TapeMode tape_mode) {
        Parameters parameters = model.make_parameters(context);
        upload_parameters(parameters, static_cast<float>(std::exp(value)));
        const auto trajectory = xayah::solver::simulate(model, context, initial_state_, std::span<const Control>{controls_}, parameters, tape_mode);
        return trajectory_loss(trajectory);
    }

} // namespace xayah::cloth::examples::stretch_stiffness_inverse
