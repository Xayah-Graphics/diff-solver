import std;
import xayah.solver;
import xayah.spring_chain.data;
import xayah.spring_chain.model;

namespace {

    template <xayah::spring_chain::Scalar T>
    class Adam {
    public:
        Adam(const std::size_t value_count, const T learning_rate) : first_moment_(value_count, T{0}), second_moment_(value_count, T{0}), learning_rate_(learning_rate) {}

        void step(const std::span<T> values, const std::span<const T> gradients) {
            ++step_;
            const T first_correction  = T{1} - std::pow(T{0.9}, static_cast<T>(step_));
            const T second_correction = T{1} - std::pow(T{0.999}, static_cast<T>(step_));
            for (std::size_t i = 0; i < values.size(); ++i) {
                first_moment_[i]         = T{0.9} * first_moment_[i] + T{0.1} * gradients[i];
                second_moment_[i]        = T{0.999} * second_moment_[i] + T{0.001} * gradients[i] * gradients[i];
                const T corrected_first  = first_moment_[i] / first_correction;
                const T corrected_second = second_moment_[i] / second_correction;
                values[i] -= learning_rate_ * corrected_first / (std::sqrt(corrected_second) + T{1e-8});
            }
        }

    private:
        std::vector<T> first_moment_;
        std::vector<T> second_moment_;
        T learning_rate_;
        std::size_t step_ = 0;
    };

    template <xayah::spring_chain::Scalar T>
    void run_initial_state_optimization(const std::string_view backend) {
        xayah::spring_chain::Configuration<T> configuration{
            .edges     = {},
            .anchors   = std::vector<std::optional<T>>(8),
            .time_step = T{0.02},
        };
        configuration.anchors[0] = T{0};
        for (std::size_t particle = 0; particle < 7; ++particle) configuration.edges.push_back({particle, particle + 1});

        xayah::spring_chain::Model<T> model(std::move(configuration));
        xayah::spring_chain::ExecutionContext<T> context = model.make_context();
        const xayah::spring_chain::Parameters<T> parameters{
            .masses       = std::vector<T>(8, T{1}),
            .stiffnesses  = std::vector<T>(7, T{20}),
            .dampings     = std::vector<T>(7, T{0.5}),
            .rest_lengths = std::vector<T>(7, T{1}),
        };
        const std::vector<xayah::spring_chain::Control<T>> controls(100, xayah::spring_chain::Control<T>{.external_forces = std::vector<T>(8, T{0})});

        xayah::spring_chain::State<T> target_initial_state{
            .positions  = std::vector<T>(8),
            .velocities = std::vector<T>(8, T{0}),
        };
        target_initial_state.positions[0] = T{0};
        for (std::size_t particle = 1; particle < 8; ++particle) {
            target_initial_state.positions[particle]  = static_cast<T>(particle) + T{0.1} * std::sin(static_cast<T>(particle));
            target_initial_state.velocities[particle] = T{0.1} * std::cos(static_cast<T>(particle));
        }
        const auto target_trajectory                               = xayah::solver::simulate(model, context, target_initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, xayah::solver::TapeMode::store_all);
        const xayah::spring_chain::State<T>& target_terminal_state = target_trajectory.states.back();

        xayah::spring_chain::State<T> initial_state{
            .positions  = std::vector<T>(8),
            .velocities = std::vector<T>(8, T{0}),
        };
        for (std::size_t particle = 0; particle < 8; ++particle) initial_state.positions[particle] = static_cast<T>(particle);

        const auto terminal_loss = [&target_terminal_state](const auto& trajectory) {
            T loss = T{0};
            for (std::size_t particle = 1; particle < 8; ++particle) {
                const T position_difference = trajectory.states.back().positions[particle] - target_terminal_state.positions[particle];
                const T velocity_difference = trajectory.states.back().velocities[particle] - target_terminal_state.velocities[particle];
                loss += position_difference * position_difference + velocity_difference * velocity_difference;
            }
            return loss / T{14};
        };
        const auto make_trajectory_adjoint = [&model, &target_terminal_state](const auto& trajectory) {
            xayah::solver::TrajectoryAdjoint<xayah::spring_chain::StateAdjoint<T>> trajectory_adjoint{.states = {}};
            trajectory_adjoint.states.reserve(trajectory.states.size());
            for (std::size_t step = 0; step < trajectory.states.size(); ++step) trajectory_adjoint.states.push_back(model.make_state_adjoint());
            for (std::size_t particle = 1; particle < 8; ++particle) {
                trajectory_adjoint.states.back().positions[particle]  = (trajectory.states.back().positions[particle] - target_terminal_state.positions[particle]) / T{7};
                trajectory_adjoint.states.back().velocities[particle] = (trajectory.states.back().velocities[particle] - target_terminal_state.velocities[particle]) / T{7};
            }
            return trajectory_adjoint;
        };

        xayah::spring_chain::StateTangent<T> initial_state_tangent = model.make_state_tangent();
        T direction_norm_squared                                   = T{0};
        for (std::size_t particle = 1; particle < 8; ++particle) {
            initial_state_tangent.positions[particle]  = std::sin(static_cast<T>(particle));
            initial_state_tangent.velocities[particle] = std::cos(static_cast<T>(particle));
            direction_norm_squared += initial_state_tangent.positions[particle] * initial_state_tangent.positions[particle] + initial_state_tangent.velocities[particle] * initial_state_tangent.velocities[particle];
        }
        const T inverse_direction_norm = T{1} / std::sqrt(direction_norm_squared);
        for (std::size_t particle = 1; particle < 8; ++particle) {
            initial_state_tangent.positions[particle] *= inverse_direction_norm;
            initial_state_tangent.velocities[particle] *= inverse_direction_norm;
        }
        std::vector<xayah::spring_chain::ControlTangent<T>> control_tangents;
        control_tangents.reserve(controls.size());
        for (std::size_t step = 0; step < controls.size(); ++step) control_tangents.push_back(model.make_control_tangent());
        const xayah::spring_chain::ParameterTangent<T> parameter_tangent = model.make_parameter_tangent();
        const T epsilon                                                  = std::same_as<T, float> ? T{1e-2} : T{1e-5};

        std::println("\n=== {} initial-state optimization ===", backend);
        std::println("  gradient checks:");
        for (const xayah::solver::TapeMode tape_mode : std::array{xayah::solver::TapeMode::store_all, xayah::solver::TapeMode::recompute_step_cache}) {
            const auto trajectory                        = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, tape_mode);
            const auto trajectory_adjoint                = make_trajectory_adjoint(trajectory);
            const auto gradients                         = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, trajectory_adjoint);
            const auto trajectory_tangent                = xayah::solver::jvp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, initial_state_tangent, std::span<const xayah::spring_chain::ControlTangent<T>>{control_tangents}, parameter_tangent);
            xayah::spring_chain::State<T> positive_state = initial_state;
            xayah::spring_chain::State<T> negative_state = initial_state;
            for (std::size_t particle = 1; particle < 8; ++particle) {
                positive_state.positions[particle] += epsilon * initial_state_tangent.positions[particle];
                positive_state.velocities[particle] += epsilon * initial_state_tangent.velocities[particle];
                negative_state.positions[particle] -= epsilon * initial_state_tangent.positions[particle];
                negative_state.velocities[particle] -= epsilon * initial_state_tangent.velocities[particle];
            }
            const auto positive_trajectory = xayah::solver::simulate(model, context, positive_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, tape_mode);
            const auto negative_trajectory = xayah::solver::simulate(model, context, negative_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, tape_mode);
            const T finite_difference      = (terminal_loss(positive_trajectory) - terminal_loss(negative_trajectory)) / (T{2} * epsilon);
            T vjp_directional_derivative   = T{0};
            T jvp_inner_product            = T{0};
            T vjp_inner_product            = T{0};
            for (std::size_t particle = 1; particle < 8; ++particle) {
                vjp_directional_derivative += gradients.initial_state.positions[particle] * initial_state_tangent.positions[particle] + gradients.initial_state.velocities[particle] * initial_state_tangent.velocities[particle];
                vjp_inner_product += gradients.initial_state.positions[particle] * initial_state_tangent.positions[particle] + gradients.initial_state.velocities[particle] * initial_state_tangent.velocities[particle];
            }
            for (std::size_t step = 0; step < trajectory_adjoint.states.size(); ++step) {
                for (std::size_t particle = 1; particle < 8; ++particle) jvp_inner_product += trajectory_adjoint.states[step].positions[particle] * trajectory_tangent.states[step].positions[particle] + trajectory_adjoint.states[step].velocities[particle] * trajectory_tangent.states[step].velocities[particle];
            }
            const T finite_difference_error = std::abs(finite_difference - vjp_directional_derivative) / std::max({T{1}, std::abs(finite_difference), std::abs(vjp_directional_derivative)});
            const T adjoint_identity_error  = std::abs(jvp_inner_product - vjp_inner_product) / std::max({T{1}, std::abs(jvp_inner_product), std::abs(vjp_inner_product)});
            std::println("    {}: finite_difference={:.9e}, vjp_direction={:.9e}, relative_error={:.3e}", tape_mode == xayah::solver::TapeMode::store_all ? "store_all" : "recompute_step_cache", finite_difference, vjp_directional_derivative, finite_difference_error);
            std::println("      jvp_inner={:.9e}, vjp_inner={:.9e}, relative_error={:.3e}", jvp_inner_product, vjp_inner_product, adjoint_identity_error);
        }

        Adam<T> position_optimizer(7, T{0.03});
        Adam<T> velocity_optimizer(7, T{0.03});
        T initial_loss = T{0};
        for (std::size_t iteration = 0; iteration < 500; ++iteration) {
            const auto trajectory         = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, xayah::solver::TapeMode::store_all);
            const T loss                  = terminal_loss(trajectory);
            const auto trajectory_adjoint = make_trajectory_adjoint(trajectory);
            const auto gradients          = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, trajectory_adjoint);
            if (iteration == 0) initial_loss = loss;
            if (iteration % 50 == 0) std::println("  iteration {:3}: loss={:.9e}, terminal_rmse={:.9e}", iteration, loss, std::sqrt(loss));
            position_optimizer.step(std::span<T>{initial_state.positions}.subspan(1), std::span<const T>{gradients.initial_state.positions}.subspan(1));
            velocity_optimizer.step(std::span<T>{initial_state.velocities}.subspan(1), std::span<const T>{gradients.initial_state.velocities}.subspan(1));
        }

        const auto final_trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, xayah::solver::TapeMode::store_all);
        const T final_loss          = terminal_loss(final_trajectory);
        T initial_state_error       = T{0};
        for (std::size_t particle = 1; particle < 8; ++particle) {
            const T position_difference = initial_state.positions[particle] - target_initial_state.positions[particle];
            const T velocity_difference = initial_state.velocities[particle] - target_initial_state.velocities[particle];
            initial_state_error += position_difference * position_difference + velocity_difference * velocity_difference;
        }
        initial_state_error = std::sqrt(initial_state_error / T{14});
        std::println("  iteration 500: loss={:.9e}, terminal_rmse={:.9e}", final_loss, std::sqrt(final_loss));
        std::println("  recovered initial-state RMSE: {:.3e}", initial_state_error);
        std::println("  final/initial loss: {:.3e}", final_loss / initial_loss);
    }

    template <xayah::spring_chain::Scalar T>
    void run_control_trajectory_optimization(const std::string_view backend) {
        xayah::spring_chain::Configuration<T> configuration{
            .edges     = {},
            .anchors   = std::vector<std::optional<T>>(8),
            .time_step = T{0.02},
        };
        configuration.anchors[0] = T{0};
        for (std::size_t particle = 0; particle < 7; ++particle) configuration.edges.push_back({particle, particle + 1});

        xayah::spring_chain::Model<T> model(std::move(configuration));
        xayah::spring_chain::ExecutionContext<T> context = model.make_context();
        xayah::spring_chain::State<T> initial_state{
            .positions  = std::vector<T>(8),
            .velocities = std::vector<T>(8, T{0}),
        };
        for (std::size_t particle = 0; particle < 8; ++particle) initial_state.positions[particle] = static_cast<T>(particle);
        const xayah::spring_chain::Parameters<T> parameters{
            .masses       = std::vector<T>(8, T{1}),
            .stiffnesses  = std::vector<T>(7, T{20}),
            .dampings     = std::vector<T>(7, T{0.5}),
            .rest_lengths = std::vector<T>(7, T{1}),
        };

        std::vector<T> target_forces(100);
        for (std::size_t step = 0; step < target_forces.size(); ++step) target_forces[step] = T{5} * std::sin(T{2} * std::numbers::pi_v<T> * static_cast<T>(step) / T{100});
        const auto make_controls = [](const std::span<const T> terminal_forces) {
            std::vector<xayah::spring_chain::Control<T>> controls(terminal_forces.size(), xayah::spring_chain::Control<T>{.external_forces = std::vector<T>(8, T{0})});
            for (std::size_t step = 0; step < controls.size(); ++step) controls[step].external_forces.back() = terminal_forces[step];
            return controls;
        };
        const std::vector<xayah::spring_chain::Control<T>> target_controls = make_controls(target_forces);
        const auto target_trajectory                                       = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{target_controls}, parameters, xayah::solver::TapeMode::store_all);
        const auto trajectory_loss                                         = [&target_trajectory](const auto& trajectory, const std::span<const T> terminal_forces) {
            T loss = T{0};
            for (std::size_t step = 1; step < trajectory.states.size(); ++step) {
                for (std::size_t particle = 1; particle < 8; ++particle) {
                    const T difference = trajectory.states[step].positions[particle] - target_trajectory.states[step].positions[particle];
                    loss += difference * difference / T{1400};
                }
            }
            for (const T force : terminal_forces) loss += T{0.5e-7} * force * force;
            return loss;
        };
        const auto make_trajectory_adjoint = [&model, &target_trajectory](const auto& trajectory) {
            xayah::solver::TrajectoryAdjoint<xayah::spring_chain::StateAdjoint<T>> trajectory_adjoint{.states = {}};
            trajectory_adjoint.states.reserve(trajectory.states.size());
            for (std::size_t step = 0; step < trajectory.states.size(); ++step) trajectory_adjoint.states.push_back(model.make_state_adjoint());
            for (std::size_t step = 1; step < trajectory.states.size(); ++step) {
                for (std::size_t particle = 1; particle < 8; ++particle) trajectory_adjoint.states[step].positions[particle] = (trajectory.states[step].positions[particle] - target_trajectory.states[step].positions[particle]) / T{700};
            }
            return trajectory_adjoint;
        };

        std::vector<T> terminal_forces(100, T{0});
        std::vector<T> direction(100);
        T direction_norm_squared = T{0};
        for (std::size_t step = 0; step < direction.size(); ++step) {
            direction[step] = std::sin(T{0.37} * static_cast<T>(step + 1));
            direction_norm_squared += direction[step] * direction[step];
        }
        const T inverse_direction_norm = T{1} / std::sqrt(direction_norm_squared);
        for (T& value : direction) value *= inverse_direction_norm;
        const xayah::spring_chain::StateTangent<T> initial_state_tangent = model.make_state_tangent();
        const xayah::spring_chain::ParameterTangent<T> parameter_tangent = model.make_parameter_tangent();
        std::vector<xayah::spring_chain::ControlTangent<T>> control_tangents;
        control_tangents.reserve(terminal_forces.size());
        for (std::size_t step = 0; step < terminal_forces.size(); ++step) {
            control_tangents.push_back(model.make_control_tangent());
            control_tangents.back().external_forces.back() = direction[step];
        }
        const T epsilon = std::same_as<T, float> ? T{1e-2} : T{1e-5};

        std::println("\n=== {} trajectory-control optimization ===", backend);
        std::println("  gradient checks:");
        for (const xayah::solver::TapeMode tape_mode : std::array{xayah::solver::TapeMode::store_all, xayah::solver::TapeMode::recompute_step_cache}) {
            const std::vector<xayah::spring_chain::Control<T>> controls = make_controls(terminal_forces);
            const auto trajectory                                       = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, tape_mode);
            const auto trajectory_adjoint                               = make_trajectory_adjoint(trajectory);
            const auto gradients                                        = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, trajectory_adjoint);
            const auto trajectory_tangent                               = xayah::solver::jvp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, initial_state_tangent, std::span<const xayah::spring_chain::ControlTangent<T>>{control_tangents}, parameter_tangent);
            std::vector<T> positive_forces(terminal_forces.size());
            std::vector<T> negative_forces(terminal_forces.size());
            for (std::size_t step = 0; step < terminal_forces.size(); ++step) {
                positive_forces[step] = terminal_forces[step] + epsilon * direction[step];
                negative_forces[step] = terminal_forces[step] - epsilon * direction[step];
            }
            const std::vector<xayah::spring_chain::Control<T>> positive_controls = make_controls(positive_forces);
            const std::vector<xayah::spring_chain::Control<T>> negative_controls = make_controls(negative_forces);
            const auto positive_trajectory                                       = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{positive_controls}, parameters, tape_mode);
            const auto negative_trajectory                                       = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{negative_controls}, parameters, tape_mode);
            const T finite_difference                                            = (trajectory_loss(positive_trajectory, positive_forces) - trajectory_loss(negative_trajectory, negative_forces)) / (T{2} * epsilon);
            T vjp_directional_derivative                                         = T{0};
            T jvp_inner_product                                                  = T{0};
            T vjp_inner_product                                                  = T{0};
            for (std::size_t step = 0; step < terminal_forces.size(); ++step) {
                vjp_directional_derivative += (gradients.controls[step].external_forces.back() + T{1e-7} * terminal_forces[step]) * direction[step];
                vjp_inner_product += gradients.controls[step].external_forces.back() * direction[step];
            }
            for (std::size_t step = 0; step < trajectory_adjoint.states.size(); ++step) {
                for (std::size_t particle = 1; particle < 8; ++particle) jvp_inner_product += trajectory_adjoint.states[step].positions[particle] * trajectory_tangent.states[step].positions[particle];
            }
            const T finite_difference_error = std::abs(finite_difference - vjp_directional_derivative) / std::max({T{1}, std::abs(finite_difference), std::abs(vjp_directional_derivative)});
            const T adjoint_identity_error  = std::abs(jvp_inner_product - vjp_inner_product) / std::max({T{1}, std::abs(jvp_inner_product), std::abs(vjp_inner_product)});
            std::println("    {}: finite_difference={:.9e}, vjp_direction={:.9e}, relative_error={:.3e}", tape_mode == xayah::solver::TapeMode::store_all ? "store_all" : "recompute_step_cache", finite_difference, vjp_directional_derivative, finite_difference_error);
            std::println("      jvp_inner={:.9e}, vjp_inner={:.9e}, relative_error={:.3e}", jvp_inner_product, vjp_inner_product, adjoint_identity_error);
        }

        Adam<T> optimizer(terminal_forces.size(), T{0.05});
        T initial_loss = T{0};
        for (std::size_t iteration = 0; iteration < 600; ++iteration) {
            const std::vector<xayah::spring_chain::Control<T>> controls = make_controls(terminal_forces);
            const auto trajectory                                       = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, xayah::solver::TapeMode::store_all);
            const T loss                                                = trajectory_loss(trajectory, terminal_forces);
            const auto trajectory_adjoint                               = make_trajectory_adjoint(trajectory);
            const auto gradients                                        = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, trajectory_adjoint);
            T trajectory_error                                          = T{0};
            T control_norm                                              = T{0};
            for (std::size_t step = 1; step < trajectory.states.size(); ++step) {
                for (std::size_t particle = 1; particle < 8; ++particle) {
                    const T difference = trajectory.states[step].positions[particle] - target_trajectory.states[step].positions[particle];
                    trajectory_error += difference * difference;
                }
            }
            for (const T force : terminal_forces) control_norm += force * force;
            if (iteration == 0) initial_loss = loss;
            if (iteration % 50 == 0) std::println("  iteration {:3}: loss={:.9e}, trajectory_rmse={:.9e}, control_norm={:.9e}", iteration, loss, std::sqrt(trajectory_error / T{700}), std::sqrt(control_norm));
            std::vector<T> force_gradients(terminal_forces.size());
            for (std::size_t step = 0; step < terminal_forces.size(); ++step) force_gradients[step] = gradients.controls[step].external_forces.back() + T{1e-7} * terminal_forces[step];
            optimizer.step(terminal_forces, force_gradients);
        }

        const std::vector<xayah::spring_chain::Control<T>> final_controls = make_controls(terminal_forces);
        const auto final_trajectory                                       = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{final_controls}, parameters, xayah::solver::TapeMode::store_all);
        const T final_loss                                                = trajectory_loss(final_trajectory, terminal_forces);
        T trajectory_error                                                = T{0};
        T terminal_error                                                  = T{0};
        T control_norm                                                    = T{0};
        for (std::size_t step = 1; step < final_trajectory.states.size(); ++step) {
            for (std::size_t particle = 1; particle < 8; ++particle) {
                const T difference = final_trajectory.states[step].positions[particle] - target_trajectory.states[step].positions[particle];
                trajectory_error += difference * difference;
            }
        }
        for (std::size_t particle = 1; particle < 8; ++particle) {
            const T difference = final_trajectory.states.back().positions[particle] - target_trajectory.states.back().positions[particle];
            terminal_error += difference * difference;
        }
        for (const T force : terminal_forces) control_norm += force * force;
        std::println("  iteration 600: loss={:.9e}, trajectory_rmse={:.9e}, control_norm={:.9e}", final_loss, std::sqrt(trajectory_error / T{700}), std::sqrt(control_norm));
        std::println("  final-state position RMSE: {:.3e}", std::sqrt(terminal_error / T{7}));
        std::println("  final/initial loss: {:.3e}", final_loss / initial_loss);
    }

    template <xayah::spring_chain::Scalar T>
    void run_parameter_identification(const std::string_view backend) {
        xayah::spring_chain::Configuration<T> configuration{
            .edges     = {},
            .anchors   = std::vector<std::optional<T>>(8),
            .time_step = T{0.02},
        };
        configuration.anchors[0] = T{0};
        for (std::size_t particle = 0; particle < 7; ++particle) configuration.edges.push_back({particle, particle + 1});

        xayah::spring_chain::Model<T> model(std::move(configuration));
        xayah::spring_chain::ExecutionContext<T> context = model.make_context();
        xayah::spring_chain::State<T> initial_state{
            .positions  = std::vector<T>(8),
            .velocities = std::vector<T>(8, T{0}),
        };
        for (std::size_t particle = 0; particle < 8; ++particle) initial_state.positions[particle] = static_cast<T>(particle);
        initial_state.positions.back() += T{0.5};
        const std::vector<xayah::spring_chain::Control<T>> controls(100, xayah::spring_chain::Control<T>{.external_forces = std::vector<T>(8, T{0})});
        const xayah::spring_chain::Parameters<T> target_parameters{
            .masses       = std::vector<T>(8, T{1}),
            .stiffnesses  = std::vector<T>(7, T{20}),
            .dampings     = std::vector<T>(7, T{0.5}),
            .rest_lengths = std::vector<T>(7, T{1}),
        };
        const auto target_trajectory = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, target_parameters, xayah::solver::TapeMode::store_all);
        const auto make_parameters   = [](const std::span<const T> latent) {
            return xayah::spring_chain::Parameters<T>{
                  .masses       = std::vector<T>(8, T{1}),
                  .stiffnesses  = std::vector<T>(7, std::exp(latent[0])),
                  .dampings     = std::vector<T>(7, std::exp(latent[1])),
                  .rest_lengths = std::vector<T>(7, T{1}),
            };
        };
        const auto trajectory_loss = [&target_trajectory](const auto& trajectory) {
            T loss = T{0};
            for (std::size_t step = 1; step < trajectory.states.size(); ++step) {
                for (std::size_t particle = 1; particle < 8; ++particle) {
                    const T difference = trajectory.states[step].positions[particle] - target_trajectory.states[step].positions[particle];
                    loss += difference * difference / T{1400};
                }
            }
            return loss;
        };
        const auto make_trajectory_adjoint = [&model, &target_trajectory](const auto& trajectory) {
            xayah::solver::TrajectoryAdjoint<xayah::spring_chain::StateAdjoint<T>> trajectory_adjoint{.states = {}};
            trajectory_adjoint.states.reserve(trajectory.states.size());
            for (std::size_t step = 0; step < trajectory.states.size(); ++step) trajectory_adjoint.states.push_back(model.make_state_adjoint());
            for (std::size_t step = 1; step < trajectory.states.size(); ++step) {
                for (std::size_t particle = 1; particle < 8; ++particle) trajectory_adjoint.states[step].positions[particle] = (trajectory.states[step].positions[particle] - target_trajectory.states[step].positions[particle]) / T{700};
            }
            return trajectory_adjoint;
        };

        std::vector<T> latent{std::log(T{8}), std::log(T{2})};
        const std::array<T, 2> direction{T{0.6}, T{-0.8}};
        const xayah::spring_chain::StateTangent<T> initial_state_tangent = model.make_state_tangent();
        std::vector<xayah::spring_chain::ControlTangent<T>> control_tangents;
        control_tangents.reserve(controls.size());
        for (std::size_t step = 0; step < controls.size(); ++step) control_tangents.push_back(model.make_control_tangent());
        const T epsilon = std::same_as<T, float> ? T{1e-2} : T{1e-5};

        std::println("\n=== {} parameter identification ===", backend);
        std::println("  gradient checks:");
        for (const xayah::solver::TapeMode tape_mode : std::array{xayah::solver::TapeMode::store_all, xayah::solver::TapeMode::recompute_step_cache}) {
            const xayah::spring_chain::Parameters<T> parameters        = make_parameters(latent);
            const auto trajectory                                      = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, tape_mode);
            const auto trajectory_adjoint                              = make_trajectory_adjoint(trajectory);
            const auto gradients                                       = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, trajectory_adjoint);
            xayah::spring_chain::ParameterTangent<T> parameter_tangent = model.make_parameter_tangent();
            for (std::size_t edge = 0; edge < 7; ++edge) {
                parameter_tangent.stiffnesses[edge] = parameters.stiffnesses[edge] * direction[0];
                parameter_tangent.dampings[edge]    = parameters.dampings[edge] * direction[1];
            }
            const auto trajectory_tangent = xayah::solver::jvp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, initial_state_tangent, std::span<const xayah::spring_chain::ControlTangent<T>>{control_tangents}, parameter_tangent);
            std::vector<T> positive_latent{latent[0] + epsilon * direction[0], latent[1] + epsilon * direction[1]};
            std::vector<T> negative_latent{latent[0] - epsilon * direction[0], latent[1] - epsilon * direction[1]};
            const xayah::spring_chain::Parameters<T> positive_parameters = make_parameters(positive_latent);
            const xayah::spring_chain::Parameters<T> negative_parameters = make_parameters(negative_latent);
            const auto positive_trajectory                               = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, positive_parameters, tape_mode);
            const auto negative_trajectory                               = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, negative_parameters, tape_mode);
            const T finite_difference                                    = (trajectory_loss(positive_trajectory) - trajectory_loss(negative_trajectory)) / (T{2} * epsilon);
            T stiffness_gradient                                         = T{0};
            T damping_gradient                                           = T{0};
            T jvp_inner_product                                          = T{0};
            T vjp_inner_product                                          = T{0};
            for (std::size_t edge = 0; edge < 7; ++edge) {
                stiffness_gradient += gradients.parameters.stiffnesses[edge];
                damping_gradient += gradients.parameters.dampings[edge];
                vjp_inner_product += gradients.parameters.stiffnesses[edge] * parameter_tangent.stiffnesses[edge] + gradients.parameters.dampings[edge] * parameter_tangent.dampings[edge];
            }
            const T vjp_directional_derivative = parameters.stiffnesses[0] * stiffness_gradient * direction[0] + parameters.dampings[0] * damping_gradient * direction[1];
            for (std::size_t step = 0; step < trajectory_adjoint.states.size(); ++step) {
                for (std::size_t particle = 1; particle < 8; ++particle) jvp_inner_product += trajectory_adjoint.states[step].positions[particle] * trajectory_tangent.states[step].positions[particle];
            }
            const T finite_difference_error = std::abs(finite_difference - vjp_directional_derivative) / std::max({T{1}, std::abs(finite_difference), std::abs(vjp_directional_derivative)});
            const T adjoint_identity_error  = std::abs(jvp_inner_product - vjp_inner_product) / std::max({T{1}, std::abs(jvp_inner_product), std::abs(vjp_inner_product)});
            std::println("    {}: finite_difference={:.9e}, vjp_direction={:.9e}, relative_error={:.3e}", tape_mode == xayah::solver::TapeMode::store_all ? "store_all" : "recompute_step_cache", finite_difference, vjp_directional_derivative, finite_difference_error);
            std::println("      jvp_inner={:.9e}, vjp_inner={:.9e}, relative_error={:.3e}", jvp_inner_product, vjp_inner_product, adjoint_identity_error);
        }

        Adam<T> optimizer(latent.size(), T{0.03});
        T initial_loss = T{0};
        for (std::size_t iteration = 0; iteration < 300; ++iteration) {
            const xayah::spring_chain::Parameters<T> parameters = make_parameters(latent);
            const auto trajectory                               = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, xayah::solver::TapeMode::store_all);
            const T loss                                        = trajectory_loss(trajectory);
            const auto trajectory_adjoint                       = make_trajectory_adjoint(trajectory);
            const auto gradients                                = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, trajectory_adjoint);
            std::array<T, 2> latent_gradients{T{0}, T{0}};
            for (std::size_t edge = 0; edge < 7; ++edge) {
                latent_gradients[0] += gradients.parameters.stiffnesses[edge];
                latent_gradients[1] += gradients.parameters.dampings[edge];
            }
            latent_gradients[0] *= parameters.stiffnesses[0];
            latent_gradients[1] *= parameters.dampings[0];
            if (iteration == 0) initial_loss = loss;
            if (iteration % 50 == 0) std::println("  iteration {:3}: loss={:.9e}, stiffness={:.9f}, damping={:.9f}", iteration, loss, parameters.stiffnesses[0], parameters.dampings[0]);
            optimizer.step(latent, latent_gradients);
        }

        const xayah::spring_chain::Parameters<T> final_parameters = make_parameters(latent);
        const auto final_trajectory                               = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, final_parameters, xayah::solver::TapeMode::store_all);
        const T final_loss                                        = trajectory_loss(final_trajectory);
        std::println("  iteration 300: loss={:.9e}, stiffness={:.9f}, damping={:.9f}", final_loss, final_parameters.stiffnesses[0], final_parameters.dampings[0]);
        std::println("  stiffness error: {:.3e}", std::abs(final_parameters.stiffnesses[0] - T{20}));
        std::println("  damping error: {:.3e}", std::abs(final_parameters.dampings[0] - T{0.5}));
        std::println("  final/initial loss: {:.3e}", final_loss / initial_loss);
    }

    template <xayah::spring_chain::Scalar T>
    void run_all_tasks(const std::string_view backend) {
        run_initial_state_optimization<T>(backend);
        run_control_trajectory_optimization<T>(backend);
        run_parameter_identification<T>(backend);
    }

} // namespace

int main() {
    run_all_tasks<float>("float");
    run_all_tasks<double>("double");
    return 0;
}
