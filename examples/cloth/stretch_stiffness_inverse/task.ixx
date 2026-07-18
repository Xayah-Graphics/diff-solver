export module xayah.examples.cloth.stretch_stiffness_inverse;

import std;
import xayah.cloth.data;
import xayah.cloth.model;
import xayah.solver;

export namespace xayah::cloth::examples::stretch_stiffness_inverse {

    struct StretchStiffnessInverseOptions {
        float mass{0.05F};
        float target_stretch_stiffness{400.0F};
        float initial_stretch_stiffness{100.0F};
        float stretch_damping{1.0F};
        float bending_stiffness{5.0F};
        float bending_damping{0.1F};
        std::size_t trajectory_steps{120u};
        float adam_learning_rate{0.05F};
    };

    struct StretchStiffnessInverseMetrics {
        std::size_t iteration;
        float stretch_stiffness;
        double loss;
        double initial_loss;
        double log_stiffness_gradient;
    };

    struct StretchStiffnessGradientCheck {
        double finite_difference;
        double jvp_inner_product;
        double vjp_inner_product;
    };

    struct StretchStiffnessInverseTask final {
        StretchStiffnessInverseOptions options;
        Model model;
        ExecutionContext context;
        xayah::solver::Trajectory<::xayah::cloth::State, StepCache> target_trajectory;
        xayah::solver::Trajectory<::xayah::cloth::State, StepCache> estimated_trajectory;
        StretchStiffnessInverseMetrics metrics;

        explicit StretchStiffnessInverseTask(Configuration configuration, StretchStiffnessInverseOptions options = {});
        StretchStiffnessInverseTask(const StretchStiffnessInverseTask&) = delete;
        StretchStiffnessInverseTask(StretchStiffnessInverseTask&&) = delete;
        StretchStiffnessInverseTask& operator=(const StretchStiffnessInverseTask&) = delete;
        StretchStiffnessInverseTask& operator=(StretchStiffnessInverseTask&&) = delete;
        ~StretchStiffnessInverseTask() noexcept;

        void reset();
        void optimize_step();
        [[nodiscard]] StretchStiffnessGradientCheck check_gradient(xayah::solver::TapeMode tape_mode, float epsilon);

    private:
        void upload_parameters(Parameters& parameters, float stretch_stiffness);
        [[nodiscard]] double trajectory_loss(const xayah::solver::Trajectory<::xayah::cloth::State, StepCache>& trajectory);
        void evaluate(xayah::solver::TapeMode tape_mode);
        [[nodiscard]] double loss_at_log_stiffness(double log_stiffness, xayah::solver::TapeMode tape_mode);

        ::xayah::cloth::State initial_state_;
        std::vector<Control> controls_;
        Parameters estimated_parameters_;
        xayah::solver::TrajectoryAdjoint<StateAdjoint> trajectory_adjoint_;
        double* scalar_;
        double log_stiffness_;
        double first_moment_;
        double second_moment_;
        std::size_t adam_step_;
    };

} // namespace xayah::cloth::examples::stretch_stiffness_inverse
