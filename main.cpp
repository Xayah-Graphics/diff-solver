import std;
import xayah.solver;
import xayah.spring_chain.data;
import xayah.spring_chain.model;

namespace {

    template <xayah::spring_chain::Scalar T>
    class Adam {
    public:
        explicit Adam(const std::size_t value_count) : first_moment_(value_count, T{0}), second_moment_(value_count, T{0}) {}

        void step(std::vector<T>& values, const std::vector<T>& gradients) {
            ++step_;
            const T first_correction  = T{1} - std::pow(T{0.9}, static_cast<T>(step_));
            const T second_correction = T{1} - std::pow(T{0.999}, static_cast<T>(step_));
            for (std::size_t i = 0; i < values.size(); ++i) {
                first_moment_[i]         = T{0.9} * first_moment_[i] + T{0.1} * gradients[i];
                second_moment_[i]        = T{0.999} * second_moment_[i] + T{0.001} * gradients[i] * gradients[i];
                const T corrected_first  = first_moment_[i] / first_correction;
                const T corrected_second = second_moment_[i] / second_correction;
                values[i] -= T{0.05} * corrected_first / (std::sqrt(corrected_second) + T{1e-8});
            }
        }

    private:
        std::vector<T> first_moment_;
        std::vector<T> second_moment_;
        std::size_t step_ = 0;
    };

    template <xayah::spring_chain::Scalar T>
    void run_control_optimization(const std::string_view backend) {
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

        std::vector<T> terminal_forces(100, T{0});
        Adam<T> optimizer(terminal_forces.size());
        T initial_loss = T{0};

        const auto make_controls = [&terminal_forces] {
            std::vector<xayah::spring_chain::Control<T>> controls(terminal_forces.size(), xayah::spring_chain::Control<T>{.external_forces = std::vector<T>(8, T{0})});
            for (std::size_t step = 0; step < controls.size(); ++step) controls[step].external_forces.back() = terminal_forces[step];
            return controls;
        };
        const auto control_norm = [&terminal_forces] {
            T squared_norm = T{0};
            for (const T force : terminal_forces) squared_norm += force * force;
            return std::sqrt(squared_norm);
        };

        std::println("\n=== {} control optimization ===", backend);
        for (std::size_t iteration = 0; iteration < 300; ++iteration) {
            const std::vector<xayah::spring_chain::Control<T>> controls = make_controls();
            const auto trajectory                                       = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, xayah::solver::TapeMode::store_all);
            const T difference                                          = trajectory.states.back().positions.back() - T{8};
            T loss                                                      = T{0.5} * difference * difference;
            for (const T force : terminal_forces) loss += T{0.5e-5} * force * force;
            if (iteration == 0) initial_loss = loss;
            if (iteration % 50 == 0) std::println("  iteration {:3}: loss={:.9e}, terminal={:.9f}, control_norm={:.9f}", iteration, loss, trajectory.states.back().positions.back(), control_norm());

            xayah::solver::TrajectoryAdjoint<xayah::spring_chain::StateAdjoint<T>> trajectory_adjoint{.states = {}};
            trajectory_adjoint.states.reserve(trajectory.states.size());
            for (std::size_t step = 0; step < trajectory.states.size(); ++step) trajectory_adjoint.states.push_back(model.make_state_adjoint());
            trajectory_adjoint.states.back().positions.back() = difference;
            const auto gradients                              = xayah::solver::vjp(model, context, trajectory, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, trajectory_adjoint);
            std::vector<T> terminal_force_gradients(terminal_forces.size());
            for (std::size_t step = 0; step < terminal_forces.size(); ++step) terminal_force_gradients[step] = gradients.controls[step].external_forces.back() + T{1e-5} * terminal_forces[step];
            optimizer.step(terminal_forces, terminal_force_gradients);
        }

        const std::vector<xayah::spring_chain::Control<T>> controls = make_controls();
        const auto trajectory                                       = xayah::solver::simulate(model, context, initial_state, std::span<const xayah::spring_chain::Control<T>>{controls}, parameters, xayah::solver::TapeMode::store_all);
        const T difference                                          = trajectory.states.back().positions.back() - T{8};
        T final_loss                                                = T{0.5} * difference * difference;
        for (const T force : terminal_forces) final_loss += T{0.5e-5} * force * force;
        std::println("  iteration 300: loss={:.9e}, terminal={:.9f}, control_norm={:.9f}", final_loss, trajectory.states.back().positions.back(), control_norm());
        std::println("  final terminal error: {:.3e}", std::abs(difference));
        std::println("  final/initial loss: {:.3e}", final_loss / initial_loss);
    }

} // namespace

int main() {
    run_control_optimization<float>("float");
    run_control_optimization<double>("double");
    return 0;
}
