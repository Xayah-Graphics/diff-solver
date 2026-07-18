export module xayah.examples.cloth.wind_trajectory_optimization;

import std;
import xayah.cloth.data;
import xayah.cloth.model;
import xayah.solver;

export namespace xayah::cloth::examples::wind_trajectory_optimization {

    struct WindTrajectoryOptimizationOptions {
        float mass{0.05F};
        float stretch_stiffness{400.0F};
        float stretch_damping{1.0F};
        float bending_stiffness{5.0F};
        float bending_damping{0.1F};
        std::size_t trajectory_steps{120u};
        float adam_learning_rate{0.02F};
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
        xayah::solver::Trajectory<::xayah::cloth::State, StepCache> target_trajectory;
        xayah::solver::Trajectory<::xayah::cloth::State, StepCache> estimated_trajectory;
        std::vector<Vector3> target_keyframes;
        std::vector<Vector3> estimated_keyframes;
        WindTrajectoryOptimizationMetrics metrics;

        explicit WindTrajectoryOptimizationTask(Configuration configuration, WindTrajectoryOptimizationOptions options = {});
        WindTrajectoryOptimizationTask(const WindTrajectoryOptimizationTask&) = delete;
        WindTrajectoryOptimizationTask(WindTrajectoryOptimizationTask&&) = delete;
        WindTrajectoryOptimizationTask& operator=(const WindTrajectoryOptimizationTask&) = delete;
        WindTrajectoryOptimizationTask& operator=(WindTrajectoryOptimizationTask&&) = delete;
        ~WindTrajectoryOptimizationTask() noexcept;

        void reset();
        void optimize_step();
        [[nodiscard]] WindTrajectoryGradientCheck check_gradient(xayah::solver::TapeMode tape_mode, float epsilon);

    private:
        static constexpr std::size_t keyframe_count = 6u;
        static constexpr std::size_t variable_count = 2u * keyframe_count;

        void upload_parameters();
        void upload_keyframes(std::span<const Vector3> keyframes);
        void write_controls(std::vector<Control>& controls, std::span<const Vector3> keyframes);
        [[nodiscard]] double trajectory_loss(const xayah::solver::Trajectory<::xayah::cloth::State, StepCache>& trajectory);
        void evaluate(xayah::solver::TapeMode tape_mode);
        [[nodiscard]] double loss_at_keyframes(std::span<const Vector3> keyframes, xayah::solver::TapeMode tape_mode);

        ::xayah::cloth::State initial_state_;
        std::vector<Control> estimated_controls_;
        std::vector<Control> probe_controls_;
        Parameters parameters_;
        xayah::solver::TrajectoryAdjoint<StateAdjoint> trajectory_adjoint_;
        std::array<double, variable_count> keyframe_gradients_;
        std::array<double, variable_count> first_moments_;
        std::array<double, variable_count> second_moments_;
        float* device_keyframes_;
        double* device_keyframe_gradients_;
        double* scalar_;
        std::size_t adam_step_;
    };

} // namespace xayah::cloth::examples::wind_trajectory_optimization
