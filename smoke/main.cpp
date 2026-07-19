import std;
import xayah.smoke.data;
import xayah.smoke.model;
import xayah.smoke.operators;
import xayah.solver;

namespace {

    void require_gradient(const std::string_view name, const double finite_difference, const double jvp, const double vjp) {
        const double finite_difference_error = std::abs(finite_difference - jvp) / std::max({1.0e-12, std::abs(finite_difference), std::abs(jvp)});
        const double adjoint_error = std::abs(jvp - vjp) / std::max({1.0e-12, std::abs(jvp), std::abs(vjp)});
        if (finite_difference_error > 5.0e-3) throw std::runtime_error(std::format("{} finite-difference check failed", name));
        if (adjoint_error > 1.0e-5) throw std::runtime_error(std::format("{} adjoint check failed", name));
    }

    xayah::smoke::Configuration small_configuration() {
        return {.resolution = {6u, 8u, 6u}, .cell_size = 0.1F, .time_step = 0.01F};
    }

    xayah::smoke::Configuration periodic_configuration() {
        xayah::smoke::Configuration configuration = small_configuration();
        configuration.velocity_boundary.x_min.mode = xayah::smoke::VelocityBoundaryMode::periodic;
        configuration.velocity_boundary.x_max.mode = xayah::smoke::VelocityBoundaryMode::periodic;
        configuration.velocity_boundary.y_min.mode = xayah::smoke::VelocityBoundaryMode::fixed_value;
        configuration.velocity_boundary.y_max.mode = xayah::smoke::VelocityBoundaryMode::zero_gradient;
        configuration.velocity_boundary.z_min.mode = xayah::smoke::VelocityBoundaryMode::periodic;
        configuration.velocity_boundary.z_max.mode = xayah::smoke::VelocityBoundaryMode::periodic;
        configuration.pressure_boundary.x_min.mode = xayah::smoke::ScalarBoundaryMode::periodic;
        configuration.pressure_boundary.x_max.mode = xayah::smoke::ScalarBoundaryMode::periodic;
        configuration.pressure_boundary.y_min.mode = xayah::smoke::ScalarBoundaryMode::zero_gradient;
        configuration.pressure_boundary.y_max.mode = xayah::smoke::ScalarBoundaryMode::fixed_value;
        configuration.pressure_boundary.z_min.mode = xayah::smoke::ScalarBoundaryMode::periodic;
        configuration.pressure_boundary.z_max.mode = xayah::smoke::ScalarBoundaryMode::periodic;
        configuration.density_boundary = configuration.pressure_boundary;
        configuration.temperature_boundary = configuration.pressure_boundary;
        configuration.vorticity_confinement_enabled = false;
        return configuration;
    }

    std::vector<float> values(const std::size_t count, const float phase) {
        std::vector<float> result(count);
        for (std::size_t index = 0u; index < count; ++index) result[index] = 0.15F * std::sin(0.37F * static_cast<float>(index) + phase) + 0.08F * std::cos(0.19F * static_cast<float>(index) - phase);
        return result;
    }

    std::vector<double> adjoint_values(const std::size_t count, const float phase) {
        std::vector<double> result(count);
        for (std::size_t index = 0u; index < count; ++index) result[index] = 0.15 * std::sin(0.37 * static_cast<double>(index) + phase) + 0.08 * std::cos(0.19 * static_cast<double>(index) - phase);
        return result;
    }

    void upload(xayah::smoke::ExecutionContext& context, xayah::smoke::StaggeredVectorField& field, const float phase) {
        context.upload(values(field.x.size, phase), field.x);
        context.upload(values(field.y.size, phase + 0.7F), field.y);
        context.upload(values(field.z.size, phase + 1.4F), field.z);
    }

    void upload(xayah::smoke::ExecutionContext& context, xayah::smoke::ScalarField& field, const float phase) {
        context.upload(values(field.values.size, phase), field.values);
    }

    void upload(xayah::smoke::ExecutionContext& context, xayah::smoke::CenteredVectorField& field, const float phase) {
        upload(context, field.x, phase);
        upload(context, field.y, phase + 0.7F);
        upload(context, field.z, phase + 1.4F);
    }

    void upload(xayah::smoke::ExecutionContext& context, xayah::smoke::ScalarAdjointField& field, const float phase) {
        context.upload(adjoint_values(field.values.size, phase), field.values);
    }

    void upload(xayah::smoke::ExecutionContext& context, xayah::smoke::CenteredVectorAdjointField& field, const float phase) {
        upload(context, field.x, phase);
        upload(context, field.y, phase + 0.7F);
        upload(context, field.z, phase + 1.4F);
    }

    void upload(xayah::smoke::ExecutionContext& context, xayah::smoke::StaggeredVectorAdjointField& field, const float phase) {
        context.upload(adjoint_values(field.x.size, phase), field.x);
        context.upload(adjoint_values(field.y.size, phase + 0.7F), field.y);
        context.upload(adjoint_values(field.z.size, phase + 1.4F), field.z);
    }

    void add_direction(xayah::smoke::ExecutionContext& context, const xayah::smoke::ScalarField& value, const xayah::smoke::ScalarField& direction, const float scale, xayah::smoke::ScalarField& output) {
        std::vector<float> value_host(value.values.size);
        std::vector<float> direction_host(value.values.size);
        context.download(value.values, value_host);
        context.download(direction.values, direction_host);
        for (std::size_t index = 0u; index < value_host.size(); ++index) value_host[index] += scale * direction_host[index];
        context.upload(value_host, output.values);
    }

    void add_direction(xayah::smoke::ExecutionContext& context, const xayah::smoke::StaggeredVectorField& value, const xayah::smoke::StaggeredVectorField& direction, const float scale, xayah::smoke::StaggeredVectorField& output) {
        std::vector<float> value_host;
        std::vector<float> direction_host;
        std::vector<float> output_host;
        const std::array inputs{std::pair{&value.x, &direction.x}, std::pair{&value.y, &direction.y}, std::pair{&value.z, &direction.z}};
        const std::array outputs{&output.x, &output.y, &output.z};
        for (std::size_t component = 0u; component < inputs.size(); ++component) {
            value_host.resize(inputs[component].first->size);
            direction_host.resize(inputs[component].first->size);
            output_host.resize(inputs[component].first->size);
            context.download(*inputs[component].first, value_host);
            context.download(*inputs[component].second, direction_host);
            for (std::size_t index = 0u; index < output_host.size(); ++index) output_host[index] = value_host[index] + scale * direction_host[index];
            context.upload(output_host, *outputs[component]);
        }
    }

    void add_direction(xayah::smoke::ExecutionContext& context, const xayah::smoke::CenteredVectorField& value, const xayah::smoke::CenteredVectorField& direction, const float scale, xayah::smoke::CenteredVectorField& output) {
        add_direction(context, value.x, direction.x, scale, output.x);
        add_direction(context, value.y, direction.y, scale, output.y);
        add_direction(context, value.z, direction.z, scale, output.z);
    }

    double dot(xayah::smoke::ExecutionContext& context, const xayah::smoke::StaggeredVectorField& first, const xayah::smoke::StaggeredVectorField& second) {
        double result = 0.0;
        const std::array first_components{&first.x, &first.y, &first.z};
        const std::array second_components{&second.x, &second.y, &second.z};
        for (std::size_t component = 0u; component < first_components.size(); ++component) {
            std::vector<float> first_host(first_components[component]->size);
            std::vector<float> second_host(second_components[component]->size);
            context.download(*first_components[component], first_host);
            context.download(*second_components[component], second_host);
            for (std::size_t index = 0u; index < first_host.size(); ++index) result += static_cast<double>(first_host[index]) * second_host[index];
        }
        return result;
    }

    double dot(xayah::smoke::ExecutionContext& context, const xayah::smoke::ScalarField& first, const xayah::smoke::ScalarField& second) {
        std::vector<float> first_host(first.values.size);
        std::vector<float> second_host(second.values.size);
        context.download(first.values, first_host);
        context.download(second.values, second_host);
        double result = 0.0;
        for (std::size_t index = 0u; index < first_host.size(); ++index) result += static_cast<double>(first_host[index]) * second_host[index];
        return result;
    }

    double dot(xayah::smoke::ExecutionContext& context, const xayah::smoke::CenteredVectorField& first, const xayah::smoke::CenteredVectorField& second) {
        return dot(context, first.x, second.x) + dot(context, first.y, second.y) + dot(context, first.z, second.z);
    }

    double dot(xayah::smoke::ExecutionContext& context, const xayah::smoke::ScalarField& first, const xayah::smoke::ScalarAdjointField& second) {
        std::vector<float> first_host(first.values.size);
        std::vector<double> second_host(second.values.size);
        context.download(first.values, first_host);
        context.download(second.values, second_host);
        double result = 0.0;
        for (std::size_t index = 0u; index < first_host.size(); ++index) result += static_cast<double>(first_host[index]) * second_host[index];
        return result;
    }

    double dot(xayah::smoke::ExecutionContext& context, const xayah::smoke::StaggeredVectorField& first, const xayah::smoke::StaggeredVectorAdjointField& second) {
        double result = 0.0;
        const std::array first_components{&first.x, &first.y, &first.z};
        const std::array second_components{&second.x, &second.y, &second.z};
        for (std::size_t component = 0u; component < first_components.size(); ++component) {
            std::vector<float> first_host(first_components[component]->size);
            std::vector<double> second_host(second_components[component]->size);
            context.download(*first_components[component], first_host);
            context.download(*second_components[component], second_host);
            for (std::size_t index = 0u; index < first_host.size(); ++index) result += static_cast<double>(first_host[index]) * second_host[index];
        }
        return result;
    }

    double dot(xayah::smoke::ExecutionContext& context, const xayah::smoke::CenteredVectorField& first, const xayah::smoke::CenteredVectorAdjointField& second) {
        return dot(context, first.x, second.x) + dot(context, first.y, second.y) + dot(context, first.z, second.z);
    }

    void check_projection() {
        xayah::smoke::Configuration configuration = periodic_configuration();
        xayah::smoke::Model model(configuration);
        xayah::smoke::ExecutionContext context = model.make_context(xayah::smoke::ExecutionMode::differentiable);
        xayah::smoke::State input = model.make_state(context);
        xayah::smoke::State positive = model.make_state(context);
        xayah::smoke::State negative = model.make_state(context);
        xayah::smoke::State positive_output = model.make_state(context);
        xayah::smoke::State negative_output = model.make_state(context);
        xayah::smoke::StateTangent direction = model.make_state_tangent(context);
        xayah::smoke::StateTangent tangent_output = model.make_state_tangent(context);
        xayah::smoke::StateAdjoint seed = model.make_state_adjoint(context);
        xayah::smoke::StateAdjoint input_adjoint = model.make_state_adjoint(context);
        xayah::smoke::StateAdjoint pressure_adjoint = model.make_state_adjoint(context);
        xayah::smoke::StateAdjoint rhs_adjoint = model.make_state_adjoint(context);
        xayah::smoke::State pressure = model.make_state(context);
        xayah::smoke::State rhs = model.make_state(context);
        upload(context, input.velocity, 0.2F);
        upload(context, direction.velocity, 1.3F);
        upload(context, seed.velocity, 2.1F);
        constexpr float epsilon = 1.0e-3F;
        add_direction(context, input.velocity, direction.velocity, epsilon, positive.velocity);
        add_direction(context, input.velocity, direction.velocity, -epsilon, negative.velocity);

        xayah::smoke::PressureProjectionOperator projection;
        projection.forward(context.resource, configuration, context.domain, positive.velocity, pressure.density, rhs.density, positive_output.velocity);
        projection.forward(context.resource, configuration, context.domain, negative.velocity, pressure.density, rhs.density, negative_output.velocity);
        projection.jvp(context.resource, configuration, context.domain, direction.velocity, pressure.density, rhs.density, tangent_output.velocity);
        projection.vjp(context.resource, configuration, context.domain, seed.velocity, pressure_adjoint.density, rhs_adjoint.density, input_adjoint.velocity);
        context.synchronize();
        const double finite_difference = (dot(context, positive_output.velocity, seed.velocity) - dot(context, negative_output.velocity, seed.velocity)) / (2.0 * epsilon);
        const double jvp = dot(context, tangent_output.velocity, seed.velocity);
        const double vjp = dot(context, direction.velocity, input_adjoint.velocity);
        const double finite_difference_error = std::abs(finite_difference - jvp) / std::max({1.0e-12, std::abs(finite_difference), std::abs(jvp)});
        const double adjoint_error = std::abs(jvp - vjp) / std::max({1.0e-12, std::abs(jvp), std::abs(vjp)});
        std::println("projection: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", finite_difference, jvp, vjp, finite_difference_error, adjoint_error);
        require_gradient("projection", finite_difference, jvp, vjp);
    }

    void check_scalar_advection() {
        xayah::smoke::Configuration configuration = periodic_configuration();
        xayah::smoke::Model model(configuration);
        xayah::smoke::ExecutionContext context = model.make_context(xayah::smoke::ExecutionMode::differentiable);
        xayah::smoke::State input = model.make_state(context);
        xayah::smoke::State positive = model.make_state(context);
        xayah::smoke::State negative = model.make_state(context);
        xayah::smoke::State positive_output = model.make_state(context);
        xayah::smoke::State negative_output = model.make_state(context);
        xayah::smoke::StateTangent direction = model.make_state_tangent(context);
        xayah::smoke::StateTangent tangent_output = model.make_state_tangent(context);
        xayah::smoke::StateAdjoint seed = model.make_state_adjoint(context);
        xayah::smoke::StateAdjoint input_adjoint = model.make_state_adjoint(context);
        xayah::smoke::State collider = model.make_state(context);
        upload(context, input.density, 0.4F);
        upload(context, input.velocity, 0.9F);
        upload(context, direction.density, 1.5F);
        upload(context, direction.velocity, 2.2F);
        upload(context, seed.density, 2.8F);
        constexpr float epsilon = 1.0e-3F;
        add_direction(context, input.density, direction.density, epsilon, positive.density);
        add_direction(context, input.density, direction.density, -epsilon, negative.density);
        add_direction(context, input.velocity, direction.velocity, epsilon, positive.velocity);
        add_direction(context, input.velocity, direction.velocity, -epsilon, negative.velocity);

        xayah::smoke::ScalarAdvectionOperator advection;
        advection.forward(context.resource, configuration, context.domain, configuration.density_boundary, collider.density, positive.density, positive.velocity, positive_output.density);
        advection.forward(context.resource, configuration, context.domain, configuration.density_boundary, collider.density, negative.density, negative.velocity, negative_output.density);
        advection.jvp(context.resource, configuration, context.domain, configuration.density_boundary, input.density, direction.density, input.velocity, direction.velocity, tangent_output.density);
        advection.vjp(context.resource, configuration, context.domain, configuration.density_boundary, input.density, input.velocity, seed.density, input_adjoint.density, input_adjoint.velocity);
        context.synchronize();
        const double finite_difference = (dot(context, positive_output.density, seed.density) - dot(context, negative_output.density, seed.density)) / (2.0 * epsilon);
        const double jvp = dot(context, tangent_output.density, seed.density);
        const double vjp = dot(context, direction.density, input_adjoint.density) + dot(context, direction.velocity, input_adjoint.velocity);
        const double finite_difference_error = std::abs(finite_difference - jvp) / std::max({1.0e-12, std::abs(finite_difference), std::abs(jvp)});
        const double adjoint_error = std::abs(jvp - vjp) / std::max({1.0e-12, std::abs(jvp), std::abs(vjp)});
        std::println("scalar advection: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", finite_difference, jvp, vjp, finite_difference_error, adjoint_error);
        require_gradient("scalar advection", finite_difference, jvp, vjp);
    }

    void check_velocity_evolution() {
        xayah::smoke::Configuration configuration = periodic_configuration();
        xayah::smoke::Model model(configuration);
        xayah::smoke::ExecutionContext context = model.make_context(xayah::smoke::ExecutionMode::differentiable);
        xayah::smoke::State input = model.make_state(context);
        xayah::smoke::State positive = model.make_state(context);
        xayah::smoke::State negative = model.make_state(context);
        xayah::smoke::Control force = model.make_control(context);
        xayah::smoke::Control positive_force = model.make_control(context);
        xayah::smoke::Control negative_force = model.make_control(context);
        xayah::smoke::StateTangent direction = model.make_state_tangent(context);
        xayah::smoke::ControlTangent force_direction = model.make_control_tangent(context);
        xayah::smoke::StepCache base_cache = model.make_step_cache(context);
        xayah::smoke::StepCache positive_cache = model.make_step_cache(context);
        xayah::smoke::StepCache negative_cache = model.make_step_cache(context);
        xayah::smoke::StepCache tangent_cache = model.make_step_cache(context);
        xayah::smoke::StateAdjoint seed = model.make_state_adjoint(context);
        xayah::smoke::StateAdjoint input_adjoint = model.make_state_adjoint(context);
        xayah::smoke::ControlAdjoint force_adjoint = model.make_control_adjoint(context);
        xayah::smoke::StateAdjoint raw_adjoint = model.make_state_adjoint(context);
        xayah::smoke::StateAdjoint forced_adjoint = model.make_state_adjoint(context);
        upload(context, input.velocity, 0.5F);
        upload(context, force.external_acceleration, 1.0F);
        upload(context, direction.velocity, 1.6F);
        upload(context, force_direction.external_acceleration, 2.1F);
        upload(context, seed.velocity, 2.7F);
        constexpr float epsilon = 1.0e-3F;
        add_direction(context, input.velocity, direction.velocity, epsilon, positive.velocity);
        add_direction(context, input.velocity, direction.velocity, -epsilon, negative.velocity);
        add_direction(context, force.external_acceleration, force_direction.external_acceleration, epsilon, positive_force.external_acceleration);
        add_direction(context, force.external_acceleration, force_direction.external_acceleration, -epsilon, negative_force.external_acceleration);

        xayah::smoke::VelocityEvolutionOperator evolution;
        evolution.forward(context.resource, configuration, context.domain, input.velocity, force.external_acceleration, base_cache.forced_velocity, tangent_cache.forced_velocity, base_cache.advected_velocity);
        evolution.forward(context.resource, configuration, context.domain, positive.velocity, positive_force.external_acceleration, positive_cache.forced_velocity, tangent_cache.advected_velocity, positive_cache.advected_velocity);
        evolution.forward(context.resource, configuration, context.domain, negative.velocity, negative_force.external_acceleration, negative_cache.forced_velocity, tangent_cache.advected_velocity, negative_cache.advected_velocity);
        evolution.jvp(context.resource, configuration, context.domain, input.velocity, direction.velocity, force_direction.external_acceleration, base_cache.forced_velocity, tangent_cache.forced_velocity, tangent_cache.advected_velocity, positive.velocity);
        evolution.vjp(context.resource, configuration, context.domain, base_cache.forced_velocity, seed.velocity, raw_adjoint.velocity, forced_adjoint.velocity, input_adjoint.velocity, force_adjoint.external_acceleration);
        context.synchronize();
        const double finite_difference = (dot(context, positive_cache.advected_velocity, seed.velocity) - dot(context, negative_cache.advected_velocity, seed.velocity)) / (2.0 * epsilon);
        const double jvp = dot(context, positive.velocity, seed.velocity);
        const double vjp = dot(context, direction.velocity, input_adjoint.velocity) + dot(context, force_direction.external_acceleration, force_adjoint.external_acceleration);
        const double finite_difference_error = std::abs(finite_difference - jvp) / std::max({1.0e-12, std::abs(finite_difference), std::abs(jvp)});
        const double adjoint_error = std::abs(jvp - vjp) / std::max({1.0e-12, std::abs(jvp), std::abs(vjp)});
        std::println("velocity evolution: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", finite_difference, jvp, vjp, finite_difference_error, adjoint_error);
        require_gradient("velocity evolution", finite_difference, jvp, vjp);
    }

    void check_full_step() {
        xayah::smoke::Configuration configuration = periodic_configuration();
        xayah::smoke::Model model(configuration);
        xayah::smoke::ExecutionContext context = model.make_context(xayah::smoke::ExecutionMode::differentiable);
        xayah::smoke::State state = model.make_state(context);
        xayah::smoke::Control control = model.make_control(context);
        xayah::smoke::Control positive_control = model.make_control(context);
        xayah::smoke::Control negative_control = model.make_control(context);
        xayah::smoke::Parameters parameters = model.make_parameters(context);
        xayah::smoke::State base_output = model.make_state(context);
        xayah::smoke::State positive_output = model.make_state(context);
        xayah::smoke::State negative_output = model.make_state(context);
        xayah::smoke::StateTangent state_tangent = model.make_state_tangent(context);
        xayah::smoke::ControlTangent control_tangent = model.make_control_tangent(context);
        xayah::smoke::ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        xayah::smoke::StateTangent tangent_output = model.make_state_tangent(context);
        xayah::smoke::StateAdjoint seed = model.make_state_adjoint(context);
        xayah::smoke::StateAdjoint state_adjoint = model.make_state_adjoint(context);
        xayah::smoke::ControlAdjoint control_adjoint = model.make_control_adjoint(context);
        xayah::smoke::ParameterAdjoint parameter_adjoint = model.make_parameter_adjoint(context);
        xayah::smoke::StepCache base_cache = model.make_step_cache(context);
        xayah::smoke::StepCache positive_cache = model.make_step_cache(context);
        xayah::smoke::StepCache negative_cache = model.make_step_cache(context);
        upload(context, state.density, 0.3F);
        upload(context, state.temperature, 0.8F);
        upload(context, state.velocity, 1.2F);
        upload(context, control.density_source, 1.7F);
        upload(context, control.temperature_source, 2.1F);
        upload(context, control.external_acceleration, 2.5F);
        upload(context, control_tangent.external_acceleration, 3.0F);
        upload(context, seed.density, 3.4F);
        upload(context, seed.temperature, 3.8F);
        upload(context, seed.velocity, 4.2F);
        context.upload(0.0F, parameters.ambient_temperature);
        context.upload(-0.1F, parameters.density_buoyancy);
        context.upload(1.0F, parameters.temperature_buoyancy);
        context.upload(0.0F, parameters.vorticity_confinement);
        constexpr float epsilon = 2.0e-3F;
        add_direction(context, control.external_acceleration, control_tangent.external_acceleration, epsilon, positive_control.external_acceleration);
        add_direction(context, control.external_acceleration, control_tangent.external_acceleration, -epsilon, negative_control.external_acceleration);
        context.resource.copy_device(positive_control.density_source.values.data, control.density_source.values.data, control.density_source.values.size * sizeof(float));
        context.resource.copy_device(positive_control.temperature_source.values.data, control.temperature_source.values.data, control.temperature_source.values.size * sizeof(float));
        context.resource.copy_device(negative_control.density_source.values.data, control.density_source.values.data, control.density_source.values.size * sizeof(float));
        context.resource.copy_device(negative_control.temperature_source.values.data, control.temperature_source.values.data, control.temperature_source.values.size * sizeof(float));

        model.forward_step(state, control, parameters, base_output, base_cache, context);
        model.forward_step(state, positive_control, parameters, positive_output, positive_cache, context);
        model.forward_step(state, negative_control, parameters, negative_output, negative_cache, context);
        model.jvp_step(state, control, parameters, base_output, base_cache, state_tangent, control_tangent, parameter_tangent, tangent_output, context);
        model.vjp_step(state, control, parameters, base_output, base_cache, seed, state_adjoint, control_adjoint, parameter_adjoint, context);
        context.synchronize();
        const double positive_loss = dot(context, positive_output.density, seed.density) + dot(context, positive_output.temperature, seed.temperature) + dot(context, positive_output.velocity, seed.velocity);
        const double negative_loss = dot(context, negative_output.density, seed.density) + dot(context, negative_output.temperature, seed.temperature) + dot(context, negative_output.velocity, seed.velocity);
        const double finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon);
        const double jvp = dot(context, tangent_output.density, seed.density) + dot(context, tangent_output.temperature, seed.temperature) + dot(context, tangent_output.velocity, seed.velocity);
        const double vjp = dot(context, control_tangent.external_acceleration, control_adjoint.external_acceleration);
        const double finite_difference_error = std::abs(finite_difference - jvp) / std::max({1.0e-12, std::abs(finite_difference), std::abs(jvp)});
        const double adjoint_error = std::abs(jvp - vjp) / std::max({1.0e-12, std::abs(jvp), std::abs(vjp)});
        std::println("full step: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", finite_difference, jvp, vjp, finite_difference_error, adjoint_error);
        require_gradient("full step", finite_difference, jvp, vjp);
    }

    void check_short_trajectory(const std::size_t step_count) {
        xayah::smoke::Configuration configuration = periodic_configuration();
        xayah::smoke::Model model(configuration);
        xayah::smoke::ExecutionContext context = model.make_context(xayah::smoke::ExecutionMode::differentiable);
        xayah::smoke::State initial_state = model.make_state(context);
        xayah::smoke::Parameters parameters = model.make_parameters(context);
        context.upload(0.0F, parameters.ambient_temperature);
        context.upload(-0.1F, parameters.density_buoyancy);
        context.upload(1.0F, parameters.temperature_buoyancy);
        context.upload(0.0F, parameters.vorticity_confinement);
        std::vector<xayah::smoke::Control> controls;
        std::vector<xayah::smoke::Control> positive_controls;
        std::vector<xayah::smoke::Control> negative_controls;
        std::vector<xayah::smoke::ControlTangent> control_tangents;
        controls.reserve(step_count);
        positive_controls.reserve(step_count);
        negative_controls.reserve(step_count);
        control_tangents.reserve(step_count);
        constexpr float epsilon = 2.0e-3F;
        for (std::size_t step = 0u; step < step_count; ++step) {
            controls.push_back(model.make_control(context));
            positive_controls.push_back(model.make_control(context));
            negative_controls.push_back(model.make_control(context));
            control_tangents.push_back(model.make_control_tangent(context));
            upload(context, controls.back().density_source, 0.2F + static_cast<float>(step) * 0.1F);
            upload(context, controls.back().temperature_source, 0.6F + static_cast<float>(step) * 0.1F);
            upload(context, controls.back().external_acceleration, 1.0F + static_cast<float>(step) * 0.1F);
            upload(context, control_tangents.back().external_acceleration, 1.8F + static_cast<float>(step) * 0.1F);
            context.resource.copy_device(positive_controls.back().density_source.values.data, controls.back().density_source.values.data, controls.back().density_source.values.size * sizeof(float));
            context.resource.copy_device(positive_controls.back().temperature_source.values.data, controls.back().temperature_source.values.data, controls.back().temperature_source.values.size * sizeof(float));
            context.resource.copy_device(negative_controls.back().density_source.values.data, controls.back().density_source.values.data, controls.back().density_source.values.size * sizeof(float));
            context.resource.copy_device(negative_controls.back().temperature_source.values.data, controls.back().temperature_source.values.data, controls.back().temperature_source.values.size * sizeof(float));
            add_direction(context, controls.back().external_acceleration, control_tangents.back().external_acceleration, epsilon, positive_controls.back().external_acceleration);
            add_direction(context, controls.back().external_acceleration, control_tangents.back().external_acceleration, -epsilon, negative_controls.back().external_acceleration);
        }
        const auto trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::smoke::Control>{controls}, parameters, xayah::solver::TapeMode::store_all);
        const auto positive_trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::smoke::Control>{positive_controls}, parameters, xayah::solver::TapeMode::recompute_step_cache);
        const auto negative_trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::smoke::Control>{negative_controls}, parameters, xayah::solver::TapeMode::recompute_step_cache);
        xayah::smoke::StateTangent initial_tangent = model.make_state_tangent(context);
        xayah::smoke::ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        const auto trajectory_tangent = xayah::solver::jvp(model, context, trajectory, std::span<const xayah::smoke::Control>{controls}, parameters, initial_tangent, std::span<const xayah::smoke::ControlTangent>{control_tangents}, parameter_tangent);
        xayah::solver::TrajectoryAdjoint<xayah::smoke::StateAdjoint> trajectory_adjoint;
        trajectory_adjoint.states.reserve(step_count + 1u);
        for (std::size_t step = 0u; step <= step_count; ++step) {
            trajectory_adjoint.states.push_back(model.make_state_adjoint(context));
            if (step != 0u) upload(context, trajectory_adjoint.states.back().density, 2.5F + static_cast<float>(step) * 0.1F);
        }
        const auto gradients = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::smoke::Control>{controls}, parameters, trajectory_adjoint);
        context.synchronize();
        double positive_loss = 0.0;
        double negative_loss = 0.0;
        double jvp = 0.0;
        double vjp = 0.0;
        for (std::size_t step = 1u; step <= step_count; ++step) {
            positive_loss += dot(context, positive_trajectory.states[step].density, trajectory_adjoint.states[step].density);
            negative_loss += dot(context, negative_trajectory.states[step].density, trajectory_adjoint.states[step].density);
            jvp += dot(context, trajectory_tangent.states[step].density, trajectory_adjoint.states[step].density);
        }
        for (std::size_t step = 0u; step < step_count; ++step) vjp += dot(context, control_tangents[step].external_acceleration, gradients.controls[step].external_acceleration);
        const double finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon);
        const double finite_difference_error = std::abs(finite_difference - jvp) / std::max({1.0e-12, std::abs(finite_difference), std::abs(jvp)});
        const double adjoint_error = std::abs(jvp - vjp) / std::max({1.0e-12, std::abs(jvp), std::abs(vjp)});
        std::println("{}-step trajectory: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", step_count, finite_difference, jvp, vjp, finite_difference_error, adjoint_error);
        require_gradient(std::format("{}-step trajectory", step_count), finite_difference, jvp, vjp);
    }

    void check_vorticity() {
        xayah::smoke::Configuration configuration = small_configuration();
        configuration.vorticity_confinement_enabled = true;
        xayah::smoke::Model model(configuration);
        xayah::smoke::ExecutionContext context = model.make_context(xayah::smoke::ExecutionMode::differentiable);
        xayah::smoke::State state = model.make_state(context);
        xayah::smoke::State positive_state = model.make_state(context);
        xayah::smoke::State negative_state = model.make_state(context);
        xayah::smoke::Control control = model.make_control(context);
        xayah::smoke::Parameters parameters = model.make_parameters(context);
        xayah::smoke::Parameters positive_parameters = model.make_parameters(context);
        xayah::smoke::Parameters negative_parameters = model.make_parameters(context);
        xayah::smoke::StateTangent state_tangent = model.make_state_tangent(context);
        xayah::smoke::ControlTangent control_tangent = model.make_control_tangent(context);
        xayah::smoke::ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        xayah::smoke::StepCache base_cache = model.make_step_cache(context);
        xayah::smoke::StepCache positive_cache = model.make_step_cache(context);
        xayah::smoke::StepCache negative_cache = model.make_step_cache(context);
        xayah::smoke::StepCache tangent_cache = model.make_step_cache(context);
        xayah::smoke::ControlAdjoint seed = model.make_control_adjoint(context);
        xayah::smoke::StateAdjoint state_adjoint = model.make_state_adjoint(context);
        xayah::smoke::ControlAdjoint control_adjoint = model.make_control_adjoint(context);
        xayah::smoke::ParameterAdjoint parameter_adjoint = model.make_parameter_adjoint(context);
        xayah::smoke::VorticityAdjointCache vorticity_adjoint_scratch = model.make_vorticity_adjoint_cache(context);
        upload(context, state.velocity, 0.4F);
        upload(context, state_tangent.velocity, 1.1F);
        upload(context, seed.external_acceleration, 2.0F);
        context.upload(0.0F, parameters.ambient_temperature);
        context.upload(0.0F, parameters.density_buoyancy);
        context.upload(0.0F, parameters.temperature_buoyancy);
        context.upload(2.0F, parameters.vorticity_confinement);
        constexpr float epsilon = 1.0e-3F;
        add_direction(context, state.velocity, state_tangent.velocity, epsilon, positive_state.velocity);
        add_direction(context, state.velocity, state_tangent.velocity, -epsilon, negative_state.velocity);
        context.upload(2.0F + epsilon, positive_parameters.vorticity_confinement);
        context.upload(2.0F - epsilon, negative_parameters.vorticity_confinement);
        context.upload(0.0F, positive_parameters.ambient_temperature);
        context.upload(0.0F, positive_parameters.density_buoyancy);
        context.upload(0.0F, positive_parameters.temperature_buoyancy);
        context.upload(0.0F, negative_parameters.ambient_temperature);
        context.upload(0.0F, negative_parameters.density_buoyancy);
        context.upload(0.0F, negative_parameters.temperature_buoyancy);

        xayah::smoke::ForceAssemblyOperator force;
        force.forward(context.resource, configuration, context.domain, state.density, state.temperature, state.velocity, control, parameters, base_cache.force, base_cache.vorticity);
        force.forward(context.resource, configuration, context.domain, positive_state.density, positive_state.temperature, positive_state.velocity, control, parameters, positive_cache.force, positive_cache.vorticity);
        force.forward(context.resource, configuration, context.domain, negative_state.density, negative_state.temperature, negative_state.velocity, control, parameters, negative_cache.force, negative_cache.vorticity);
        force.jvp(context.resource, configuration, context.domain, state.density, state.temperature, state_tangent.velocity, state_tangent.density, state_tangent.temperature, control_tangent, parameters, parameter_tangent, tangent_cache.force, base_cache.vorticity, tangent_cache.vorticity);
        force.vjp(context.resource, configuration, context.domain, state.density, state.temperature, state.velocity, parameters, seed.external_acceleration, base_cache.vorticity, state_adjoint.density, state_adjoint.temperature, state_adjoint.velocity, control_adjoint, parameter_adjoint, vorticity_adjoint_scratch);
        context.synchronize();
        const double velocity_finite_difference = (dot(context, positive_cache.force, seed.external_acceleration) - dot(context, negative_cache.force, seed.external_acceleration)) / (2.0 * epsilon);
        const double velocity_jvp = dot(context, tangent_cache.force, seed.external_acceleration);
        const double velocity_vjp = dot(context, state_tangent.velocity, state_adjoint.velocity);

        xayah::smoke::ParameterTangent confinement_tangent = model.make_parameter_tangent(context);
        context.upload(1.0F, confinement_tangent.vorticity_confinement);
        xayah::smoke::StepCache confinement_tangent_cache = model.make_step_cache(context);
        force.forward(context.resource, configuration, context.domain, state.density, state.temperature, state.velocity, control, positive_parameters, positive_cache.force, positive_cache.vorticity);
        force.forward(context.resource, configuration, context.domain, state.density, state.temperature, state.velocity, control, negative_parameters, negative_cache.force, negative_cache.vorticity);
        xayah::smoke::StateTangent zero_state_tangent = model.make_state_tangent(context);
        force.jvp(context.resource, configuration, context.domain, state.density, state.temperature, zero_state_tangent.velocity, zero_state_tangent.density, zero_state_tangent.temperature, control_tangent, parameters, confinement_tangent, confinement_tangent_cache.force, base_cache.vorticity, confinement_tangent_cache.vorticity);
        context.synchronize();
        const double confinement_finite_difference = (dot(context, positive_cache.force, seed.external_acceleration) - dot(context, negative_cache.force, seed.external_acceleration)) / (2.0 * epsilon);
        const double confinement_jvp = dot(context, confinement_tangent_cache.force, seed.external_acceleration);
        std::array<double, 1u> confinement_adjoint{};
        context.download(parameter_adjoint.vorticity_confinement, confinement_adjoint);
        const double velocity_error = std::abs(velocity_jvp - velocity_vjp) / std::max({1.0e-12, std::abs(velocity_jvp), std::abs(velocity_vjp)});
        const double confinement_error = std::abs(confinement_jvp - confinement_adjoint[0]) / std::max({1.0e-12, std::abs(confinement_jvp), std::abs(confinement_adjoint[0])});
        std::println("vorticity velocity: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, jvp/vjp={:.3e}", velocity_finite_difference, velocity_jvp, velocity_vjp, velocity_error);
        std::println("vorticity coefficient: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, jvp/vjp={:.3e}", confinement_finite_difference, confinement_jvp, confinement_adjoint[0], confinement_error);
        require_gradient("vorticity velocity", velocity_finite_difference, velocity_jvp, velocity_vjp);
        require_gradient("vorticity coefficient", confinement_finite_difference, confinement_jvp, confinement_adjoint[0]);
    }

    void check_source() {
        xayah::smoke::Configuration configuration = small_configuration();
        xayah::smoke::Model model(configuration);
        xayah::smoke::ExecutionContext context = model.make_context(xayah::smoke::ExecutionMode::differentiable);
        xayah::smoke::State state = model.make_state(context);
        xayah::smoke::State positive_state = model.make_state(context);
        xayah::smoke::State negative_state = model.make_state(context);
        xayah::smoke::Control control = model.make_control(context);
        xayah::smoke::Control positive_control = model.make_control(context);
        xayah::smoke::Control negative_control = model.make_control(context);
        xayah::smoke::StateTangent state_tangent = model.make_state_tangent(context);
        xayah::smoke::ControlTangent control_tangent = model.make_control_tangent(context);
        xayah::smoke::StateTangent tangent_output = model.make_state_tangent(context);
        xayah::smoke::StateAdjoint seed = model.make_state_adjoint(context);
        xayah::smoke::StateAdjoint state_adjoint = model.make_state_adjoint(context);
        xayah::smoke::ControlAdjoint control_adjoint = model.make_control_adjoint(context);
        upload(context, state.density, 0.2F);
        upload(context, control.density_source, 0.7F);
        upload(context, state_tangent.density, 1.2F);
        upload(context, control_tangent.density_source, 1.8F);
        upload(context, seed.density, 2.4F);
        constexpr float epsilon = 1.0e-3F;
        add_direction(context, state.density, state_tangent.density, epsilon, positive_state.density);
        add_direction(context, state.density, state_tangent.density, -epsilon, negative_state.density);
        add_direction(context, control.density_source, control_tangent.density_source, epsilon, positive_control.density_source);
        add_direction(context, control.density_source, control_tangent.density_source, -epsilon, negative_control.density_source);
        xayah::smoke::SourceOperator source;
        source.forward(context.resource, configuration, context.domain, positive_state.density, positive_control.density_source, positive_state.temperature);
        source.forward(context.resource, configuration, context.domain, negative_state.density, negative_control.density_source, negative_state.temperature);
        source.jvp(context.resource, configuration, context.domain, state_tangent.density, control_tangent.density_source, tangent_output.density);
        source.vjp(context.resource, configuration, context.domain, seed.density, state_adjoint.density, control_adjoint.density_source);
        context.synchronize();
        const double finite_difference = (dot(context, positive_state.temperature, seed.density) - dot(context, negative_state.temperature, seed.density)) / (2.0 * epsilon);
        const double jvp = dot(context, tangent_output.density, seed.density);
        const double vjp = dot(context, state_tangent.density, state_adjoint.density) + dot(context, control_tangent.density_source, control_adjoint.density_source);
        const double finite_difference_error = std::abs(finite_difference - jvp) / std::max({1.0e-12, std::abs(finite_difference), std::abs(jvp)});
        const double adjoint_error = std::abs(jvp - vjp) / std::max({1.0e-12, std::abs(jvp), std::abs(vjp)});
        std::println("source: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", finite_difference, jvp, vjp, finite_difference_error, adjoint_error);
        require_gradient("source", finite_difference, jvp, vjp);
    }

    void check_buoyancy() {
        xayah::smoke::Configuration configuration = small_configuration();
        configuration.vorticity_confinement_enabled = false;
        xayah::smoke::Model model(configuration);
        xayah::smoke::ExecutionContext context = model.make_context(xayah::smoke::ExecutionMode::differentiable);
        xayah::smoke::State state = model.make_state(context);
        xayah::smoke::State positive_state = model.make_state(context);
        xayah::smoke::State negative_state = model.make_state(context);
        xayah::smoke::Control control = model.make_control(context);
        xayah::smoke::Control positive_control = model.make_control(context);
        xayah::smoke::Control negative_control = model.make_control(context);
        xayah::smoke::Parameters parameters = model.make_parameters(context);
        xayah::smoke::Parameters positive_parameters = model.make_parameters(context);
        xayah::smoke::Parameters negative_parameters = model.make_parameters(context);
        xayah::smoke::StateTangent state_tangent = model.make_state_tangent(context);
        xayah::smoke::ControlTangent control_tangent = model.make_control_tangent(context);
        xayah::smoke::ParameterTangent parameter_tangent = model.make_parameter_tangent(context);
        xayah::smoke::StepCache positive_cache = model.make_step_cache(context);
        xayah::smoke::StepCache negative_cache = model.make_step_cache(context);
        xayah::smoke::StepCache base_cache = model.make_step_cache(context);
        xayah::smoke::StepCache tangent_cache = model.make_step_cache(context);
        xayah::smoke::ControlAdjoint seed = model.make_control_adjoint(context);
        xayah::smoke::StateAdjoint state_adjoint = model.make_state_adjoint(context);
        xayah::smoke::ControlAdjoint control_adjoint = model.make_control_adjoint(context);
        xayah::smoke::ParameterAdjoint parameter_adjoint = model.make_parameter_adjoint(context);
        xayah::smoke::VorticityAdjointCache vorticity_adjoint_scratch = model.make_vorticity_adjoint_cache(context);
        upload(context, state.density, 0.3F);
        upload(context, state.temperature, 0.8F);
        upload(context, control.external_acceleration, 1.3F);
        upload(context, state_tangent.density, 1.8F);
        upload(context, state_tangent.temperature, 2.3F);
        upload(context, control_tangent.external_acceleration, 2.8F);
        upload(context, seed.external_acceleration, 3.3F);
        context.upload(0.1F, parameters.ambient_temperature);
        context.upload(-0.2F, parameters.density_buoyancy);
        context.upload(0.9F, parameters.temperature_buoyancy);
        context.upload(0.0F, parameters.vorticity_confinement);
        context.upload(0.4F, parameter_tangent.ambient_temperature);
        context.upload(-0.3F, parameter_tangent.density_buoyancy);
        context.upload(0.2F, parameter_tangent.temperature_buoyancy);
        constexpr float epsilon = 1.0e-3F;
        add_direction(context, state.density, state_tangent.density, epsilon, positive_state.density);
        add_direction(context, state.density, state_tangent.density, -epsilon, negative_state.density);
        add_direction(context, state.temperature, state_tangent.temperature, epsilon, positive_state.temperature);
        add_direction(context, state.temperature, state_tangent.temperature, -epsilon, negative_state.temperature);
        add_direction(context, control.external_acceleration, control_tangent.external_acceleration, epsilon, positive_control.external_acceleration);
        add_direction(context, control.external_acceleration, control_tangent.external_acceleration, -epsilon, negative_control.external_acceleration);
        context.upload(0.1F + 0.4F * epsilon, positive_parameters.ambient_temperature);
        context.upload(-0.2F - 0.3F * epsilon, positive_parameters.density_buoyancy);
        context.upload(0.9F + 0.2F * epsilon, positive_parameters.temperature_buoyancy);
        context.upload(0.1F - 0.4F * epsilon, negative_parameters.ambient_temperature);
        context.upload(-0.2F + 0.3F * epsilon, negative_parameters.density_buoyancy);
        context.upload(0.9F - 0.2F * epsilon, negative_parameters.temperature_buoyancy);
        context.upload(0.0F, positive_parameters.vorticity_confinement);
        context.upload(0.0F, negative_parameters.vorticity_confinement);
        xayah::smoke::ForceAssemblyOperator force;
        force.forward(context.resource, configuration, context.domain, state.density, state.temperature, state.velocity, control, parameters, base_cache.force, base_cache.vorticity);
        force.forward(context.resource, configuration, context.domain, positive_state.density, positive_state.temperature, positive_state.velocity, positive_control, positive_parameters, positive_cache.force, positive_cache.vorticity);
        force.forward(context.resource, configuration, context.domain, negative_state.density, negative_state.temperature, negative_state.velocity, negative_control, negative_parameters, negative_cache.force, negative_cache.vorticity);
        force.jvp(context.resource, configuration, context.domain, state.density, state.temperature, state_tangent.velocity, state_tangent.density, state_tangent.temperature, control_tangent, parameters, parameter_tangent, tangent_cache.force, base_cache.vorticity, tangent_cache.vorticity);
        force.vjp(context.resource, configuration, context.domain, state.density, state.temperature, state.velocity, parameters, seed.external_acceleration, base_cache.vorticity, state_adjoint.density, state_adjoint.temperature, state_adjoint.velocity, control_adjoint, parameter_adjoint, vorticity_adjoint_scratch);
        context.synchronize();
        const double finite_difference = (dot(context, positive_cache.force, seed.external_acceleration) - dot(context, negative_cache.force, seed.external_acceleration)) / (2.0 * epsilon);
        const double jvp = dot(context, tangent_cache.force, seed.external_acceleration);
        std::array<double, 1u> ambient_adjoint{}, density_buoyancy_adjoint{}, temperature_buoyancy_adjoint{};
        context.download(parameter_adjoint.ambient_temperature, ambient_adjoint);
        context.download(parameter_adjoint.density_buoyancy, density_buoyancy_adjoint);
        context.download(parameter_adjoint.temperature_buoyancy, temperature_buoyancy_adjoint);
        const double vjp = dot(context, state_tangent.density, state_adjoint.density) + dot(context, state_tangent.temperature, state_adjoint.temperature) + dot(context, control_tangent.external_acceleration, control_adjoint.external_acceleration) + 0.4 * ambient_adjoint[0] - 0.3 * density_buoyancy_adjoint[0] + 0.2 * temperature_buoyancy_adjoint[0];
        const double finite_difference_error = std::abs(finite_difference - jvp) / std::max({1.0e-12, std::abs(finite_difference), std::abs(jvp)});
        const double adjoint_error = std::abs(jvp - vjp) / std::max({1.0e-12, std::abs(jvp), std::abs(vjp)});
        std::println("buoyancy: fd={:.9e}, jvp={:.9e}, vjp={:.9e}, fd/jvp={:.3e}, jvp/vjp={:.3e}", finite_difference, jvp, vjp, finite_difference_error, adjoint_error);
        require_gradient("buoyancy", finite_difference, jvp, vjp);
    }

} // namespace

int main() {
    check_source();
    check_buoyancy();
    check_projection();
    check_scalar_advection();
    check_velocity_evolution();
    check_full_step();
    check_short_trajectory(10u);
    check_vorticity();
    return 0;
}
