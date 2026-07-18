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

    class WindTrajectoryOptimizationTask final {
    public:
        struct State;

        explicit WindTrajectoryOptimizationTask(Configuration configuration, WindTrajectoryOptimizationOptions options = {});
        WindTrajectoryOptimizationTask(const WindTrajectoryOptimizationTask&) = delete;
        WindTrajectoryOptimizationTask(WindTrajectoryOptimizationTask&&) noexcept;
        WindTrajectoryOptimizationTask& operator=(const WindTrajectoryOptimizationTask&) = delete;
        WindTrajectoryOptimizationTask& operator=(WindTrajectoryOptimizationTask&&) noexcept;
        ~WindTrajectoryOptimizationTask() noexcept;

        void reset();
        void optimize_step();
        [[nodiscard]] WindTrajectoryGradientCheck check_gradient(xayah::solver::TapeMode tape_mode, float epsilon);
        [[nodiscard]] const WindTrajectoryOptimizationOptions& options() const;
        [[nodiscard]] const WindTrajectoryOptimizationMetrics& metrics() const;
        [[nodiscard]] const Model& model() const;
        [[nodiscard]] ExecutionContext& context();
        [[nodiscard]] std::size_t trajectory_state_count() const;
        [[nodiscard]] const ::xayah::cloth::State& target_state(std::size_t step) const;
        [[nodiscard]] const ::xayah::cloth::State& estimated_state(std::size_t step) const;
        [[nodiscard]] std::span<const Vector3> target_wind_keyframes() const;
        [[nodiscard]] std::span<const Vector3> estimated_wind_keyframes() const;
        [[nodiscard]] Vector3 target_wind(std::size_t control_step) const;
        [[nodiscard]] Vector3 estimated_wind(std::size_t control_step) const;

    private:
        std::unique_ptr<State> state_;
    };

} // namespace xayah::cloth::examples::wind_trajectory_optimization
