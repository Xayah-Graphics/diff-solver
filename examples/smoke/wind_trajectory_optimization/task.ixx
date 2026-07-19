export module xayah.examples.smoke.wind_trajectory_optimization;

import std;
import xayah.smoke.data;
import xayah.smoke.model;
import xayah.solver;

export namespace xayah::smoke::examples::wind_trajectory_optimization {

    struct WindTrajectoryOptimizationOptions {
        std::size_t trajectory_steps{90u};
        float density_source_rate{3.0F};
        float temperature_source_rate{6.0F};
        float ambient_temperature{};
        float density_buoyancy{-0.1F};
        float temperature_buoyancy{1.0F};
        float adam_learning_rate{0.05F};
    };

    struct WindTrajectoryOptimizationMetrics {
        std::size_t iteration;
        double loss;
        double initial_loss;
        double loss_ratio;
        double keyframe_relative_error;
        double gradient_norm;
    };

    struct WindTrajectoryGradientCheck {
        double finite_difference;
        double jvp_inner_product;
        double vjp_inner_product;
    };

    struct WindTrajectoryOptimizationTask final {
        WindTrajectoryOptimizationOptions options;
        Model model;
        ExecutionContext context;
        solver::Trajectory<State, StepCache> target_trajectory;
        solver::Trajectory<State, StepCache> estimated_trajectory;
        std::vector<Vector3> target_keyframes;
        std::vector<Vector3> estimated_keyframes;
        WindTrajectoryOptimizationMetrics metrics;

        explicit WindTrajectoryOptimizationTask(WindTrajectoryOptimizationOptions options = {});
        WindTrajectoryOptimizationTask(const WindTrajectoryOptimizationTask&) = delete;
        WindTrajectoryOptimizationTask(WindTrajectoryOptimizationTask&&) = delete;
        WindTrajectoryOptimizationTask& operator=(const WindTrajectoryOptimizationTask&) = delete;
        WindTrajectoryOptimizationTask& operator=(WindTrajectoryOptimizationTask&&) = delete;
        ~WindTrajectoryOptimizationTask() noexcept;

        void reset();
        void optimize_step();
        [[nodiscard]] WindTrajectoryGradientCheck check_gradient(solver::TapeMode tape_mode, float epsilon);

    private:
        static constexpr std::size_t keyframe_count = 6u;
        static constexpr std::size_t variable_count = 2u * keyframe_count;

        void upload_parameters();
        void upload_keyframes(std::span<const Vector3> keyframes);
        void write_controls(std::vector<Control>& controls, std::span<const Vector3> keyframes);
        void write_control_tangents(std::vector<ControlTangent>& controls, std::span<const Vector3> keyframes);
        [[nodiscard]] double trajectory_loss(const solver::Trajectory<State, StepCache>& trajectory);
        void evaluate(solver::TapeMode tape_mode);
        [[nodiscard]] double loss_at_keyframes(std::span<const Vector3> keyframes, solver::TapeMode tape_mode);

        State initial_state_;
        std::vector<Control> estimated_controls_;
        std::vector<Control> probe_controls_;
        Parameters parameters_;
        solver::TrajectoryAdjoint<StateAdjoint> trajectory_adjoint_;
        std::array<double, variable_count> keyframe_gradients_;
        std::array<double, variable_count> first_moments_;
        std::array<double, variable_count> second_moments_;
        float* device_keyframes_;
        double* device_keyframe_gradients_;
        double* scalar_;
        std::vector<double> inverse_target_frame_energies_;
        std::size_t adam_step_;
    };

} // namespace xayah::smoke::examples::wind_trajectory_optimization
