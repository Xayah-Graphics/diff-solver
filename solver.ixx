export module xayah.solver;

import std;

export namespace xayah::solver {

    enum class TapeMode {
        store_all,
        recompute_step_cache,
    };

    template <typename State, typename StepCache>
    struct Trajectory {
        TapeMode tape_mode;
        std::vector<State> states;
        std::vector<StepCache> step_caches;
    };

    template <typename StateTangent>
    struct TrajectoryTangent {
        std::vector<StateTangent> states;
    };

    template <typename StateAdjoint>
    struct TrajectoryAdjoint {
        std::vector<StateAdjoint> states;
    };

    template <typename StateAdjoint, typename ControlAdjoint, typename ParameterAdjoint>
    struct GradientResult {
        StateAdjoint initial_state;
        std::vector<ControlAdjoint> controls;
        ParameterAdjoint parameters;
    };

    template <typename Model, typename Context, typename State, typename Control, typename Parameters>
    concept ForwardModel = requires(const Model& model, Context& context, const State& state, const Control& control, const Parameters& parameters, decltype(std::declval<const Model&>().make_state(context))& next_state, decltype(std::declval<const Model&>().make_step_cache(context))& step_cache) {
        { model.make_state(context) } -> std::same_as<State>;
        model.make_step_cache(context);
        { model.copy_state(state, next_state, context) } -> std::same_as<void>;
        { model.forward_step(state, control, parameters, next_state, step_cache, context) } -> std::same_as<void>;
    };

    template <typename Model, typename Context, typename State, typename Control, typename Parameters>
    concept JvpModel = ForwardModel<Model, Context, State, Control, Parameters> && requires(const Model& model, Context& context, const State& state, const Control& control, const Parameters& parameters, const State& next_state, const decltype(std::declval<const Model&>().make_step_cache(context))& step_cache, const decltype(std::declval<const Model&>().make_state_tangent(context))& state_tangent, const decltype(std::declval<const Model&>().make_control_tangent(context))& control_tangent, const decltype(std::declval<const Model&>().make_parameter_tangent(context))& parameter_tangent, decltype(std::declval<const Model&>().make_state_tangent(context))& next_state_tangent) {
        model.make_state_tangent(context);
        model.make_control_tangent(context);
        model.make_parameter_tangent(context);
        { model.copy_state_tangent(state_tangent, next_state_tangent, context) } -> std::same_as<void>;
        { model.jvp_step(state, control, parameters, next_state, step_cache, state_tangent, control_tangent, parameter_tangent, next_state_tangent, context) } -> std::same_as<void>;
    };

    template <typename Model, typename Context, typename State, typename Control, typename Parameters>
    concept VjpModel = ForwardModel<Model, Context, State, Control, Parameters> && requires(const Model& model, Context& context, const State& state, const Control& control, const Parameters& parameters, const State& next_state, const decltype(std::declval<const Model&>().make_step_cache(context))& step_cache, const decltype(std::declval<const Model&>().make_state_adjoint(context))& next_state_adjoint, decltype(std::declval<const Model&>().make_state_adjoint(context))& previous_state_adjoint, decltype(std::declval<const Model&>().make_control_adjoint(context))& control_adjoint, decltype(std::declval<const Model&>().make_parameter_adjoint(context))& parameter_adjoint) {
        model.make_state_adjoint(context);
        model.make_control_adjoint(context);
        model.make_parameter_adjoint(context);
        { model.copy_state_adjoint(next_state_adjoint, previous_state_adjoint, context) } -> std::same_as<void>;
        { model.accumulate_state_adjoint(next_state_adjoint, previous_state_adjoint, context) } -> std::same_as<void>;
        { model.vjp_step(state, control, parameters, next_state, step_cache, next_state_adjoint, previous_state_adjoint, control_adjoint, parameter_adjoint, context) } -> std::same_as<void>;
    };

    template <typename Model, typename Context, typename State, typename Control, typename Parameters>
        requires ForwardModel<Model, Context, State, Control, Parameters>
    [[nodiscard]] auto simulate(const Model& model, Context& context, const State& initial_state, const std::span<const Control> controls, const Parameters& parameters, const TapeMode tape_mode) -> Trajectory<State, decltype(model.make_step_cache(context))> {
        Trajectory<State, decltype(model.make_step_cache(context))> trajectory{
            .tape_mode   = tape_mode,
            .states      = {},
            .step_caches = {},
        };
        trajectory.states.reserve(controls.size() + 1);
        State copied_initial_state = model.make_state(context);
        model.copy_state(initial_state, copied_initial_state, context);
        trajectory.states.push_back(std::move(copied_initial_state));
        if (tape_mode == TapeMode::store_all) trajectory.step_caches.reserve(controls.size());

        for (const Control& control : controls) {
            State next_state = model.make_state(context);
            auto step_cache  = model.make_step_cache(context);
            model.forward_step(trajectory.states.back(), control, parameters, next_state, step_cache, context);
            trajectory.states.push_back(std::move(next_state));
            if (tape_mode == TapeMode::store_all) trajectory.step_caches.push_back(std::move(step_cache));
        }
        return trajectory;
    }

    template <typename Model, typename Context, typename State, typename StepCache, typename Control, typename Parameters>
        requires JvpModel<Model, Context, State, Control, Parameters>
    [[nodiscard]] auto jvp(const Model& model, Context& context, const Trajectory<State, StepCache>& trajectory, const std::span<const Control> controls, const Parameters& parameters, const decltype(model.make_state_tangent(context))& initial_state_tangent, const std::span<const decltype(model.make_control_tangent(context))> control_tangents, const decltype(model.make_parameter_tangent(context))& parameter_tangent) -> TrajectoryTangent<decltype(model.make_state_tangent(context))> {
        TrajectoryTangent<decltype(model.make_state_tangent(context))> result{.states = {}};
        result.states.reserve(trajectory.states.size());
        auto copied_initial_state_tangent = model.make_state_tangent(context);
        model.copy_state_tangent(initial_state_tangent, copied_initial_state_tangent, context);
        result.states.push_back(std::move(copied_initial_state_tangent));

        for (std::size_t step = 0; step < controls.size(); ++step) {
            auto recomputed_cache = model.make_step_cache(context);
            const StepCache* step_cache;
            if (trajectory.tape_mode == TapeMode::store_all) {
                step_cache = &trajectory.step_caches[step];
            } else {
                State recomputed_state = model.make_state(context);
                model.forward_step(trajectory.states[step], controls[step], parameters, recomputed_state, recomputed_cache, context);
                step_cache = &recomputed_cache;
            }

            auto next_state_tangent = model.make_state_tangent(context);
            model.jvp_step(trajectory.states[step], controls[step], parameters, trajectory.states[step + 1], *step_cache, result.states.back(), control_tangents[step], parameter_tangent, next_state_tangent, context);
            result.states.push_back(std::move(next_state_tangent));
        }
        return result;
    }

    template <typename Model, typename Context, typename State, typename StepCache, typename Control, typename Parameters, typename StateAdjoint>
        requires VjpModel<Model, Context, State, Control, Parameters>
    [[nodiscard]] auto vjp(const Model& model, Context& context, const Trajectory<State, StepCache>& trajectory, const std::span<const Control> controls, const Parameters& parameters, const TrajectoryAdjoint<StateAdjoint>& trajectory_adjoint) -> GradientResult<StateAdjoint, decltype(model.make_control_adjoint(context)), decltype(model.make_parameter_adjoint(context))> {
        GradientResult<StateAdjoint, decltype(model.make_control_adjoint(context)), decltype(model.make_parameter_adjoint(context))> result{
            .initial_state = model.make_state_adjoint(context),
            .controls      = {},
            .parameters    = model.make_parameter_adjoint(context),
        };
        result.controls.reserve(controls.size());
        for (std::size_t step = 0; step < controls.size(); ++step) result.controls.push_back(model.make_control_adjoint(context));

        StateAdjoint running_state_adjoint = model.make_state_adjoint(context);
        model.copy_state_adjoint(trajectory_adjoint.states.back(), running_state_adjoint, context);
        for (std::size_t step = controls.size(); step-- > 0;) {
            auto recomputed_cache = model.make_step_cache(context);
            const StepCache* step_cache;
            if (trajectory.tape_mode == TapeMode::store_all) {
                step_cache = &trajectory.step_caches[step];
            } else {
                State recomputed_state = model.make_state(context);
                model.forward_step(trajectory.states[step], controls[step], parameters, recomputed_state, recomputed_cache, context);
                step_cache = &recomputed_cache;
            }

            StateAdjoint previous_state_adjoint = model.make_state_adjoint(context);
            model.vjp_step(trajectory.states[step], controls[step], parameters, trajectory.states[step + 1], *step_cache, running_state_adjoint, previous_state_adjoint, result.controls[step], result.parameters, context);
            model.accumulate_state_adjoint(trajectory_adjoint.states[step], previous_state_adjoint, context);
            running_state_adjoint = std::move(previous_state_adjoint);
        }
        result.initial_state = std::move(running_state_adjoint);
        return result;
    }

} // namespace xayah::solver
