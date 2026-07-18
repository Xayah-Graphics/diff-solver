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

    class StretchStiffnessInverseTask final {
    public:
        struct State;

        explicit StretchStiffnessInverseTask(Configuration configuration, StretchStiffnessInverseOptions options = {});
        StretchStiffnessInverseTask(const StretchStiffnessInverseTask&) = delete;
        StretchStiffnessInverseTask(StretchStiffnessInverseTask&&) noexcept;
        StretchStiffnessInverseTask& operator=(const StretchStiffnessInverseTask&) = delete;
        StretchStiffnessInverseTask& operator=(StretchStiffnessInverseTask&&) noexcept;
        ~StretchStiffnessInverseTask() noexcept;

        void reset();
        void optimize_step();
        [[nodiscard]] StretchStiffnessGradientCheck check_gradient(xayah::solver::TapeMode tape_mode, float epsilon);
        [[nodiscard]] const StretchStiffnessInverseOptions& options() const;
        [[nodiscard]] const StretchStiffnessInverseMetrics& metrics() const;
        [[nodiscard]] const Model& model() const;
        [[nodiscard]] ExecutionContext& context();
        [[nodiscard]] std::size_t trajectory_state_count() const;
        [[nodiscard]] const ::xayah::cloth::State& target_state(std::size_t step) const;
        [[nodiscard]] const ::xayah::cloth::State& estimated_state(std::size_t step) const;

    private:
        std::unique_ptr<State> state_;
    };

} // namespace xayah::cloth::examples::stretch_stiffness_inverse
