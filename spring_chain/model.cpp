module xayah.spring_chain.model;

import std;
import xayah.spring_chain.data;
import xayah.spring_chain.operators;

namespace xayah::spring_chain {

    template <Scalar T>
    ExecutionContext<T>::ExecutionContext(const std::size_t particle_count) :
        integrated_state{
            .positions  = std::vector<T>(particle_count, T{0}),
            .velocities = std::vector<T>(particle_count, T{0}),
        },
        force_tangent{.values = std::vector<T>(particle_count, T{0})},
        integrated_state_tangent{
            .positions  = std::vector<T>(particle_count, T{0}),
            .velocities = std::vector<T>(particle_count, T{0}),
        },
        force_adjoint{.values = std::vector<T>(particle_count, T{0})},
        integrated_state_adjoint{
            .positions  = std::vector<T>(particle_count, T{0}),
            .velocities = std::vector<T>(particle_count, T{0}),
        } {}

    template struct ExecutionContext<float>;
    template struct ExecutionContext<double>;

    template <Scalar T>
    Model<T>::Model(Configuration<T> configuration) : configuration_(std::move(configuration)) {}

    template <Scalar T>
    const Configuration<T>& Model<T>::configuration() const {
        return configuration_;
    }

    template <Scalar T>
    ExecutionContext<T> Model<T>::make_context() const {
        return ExecutionContext<T>(configuration_.anchors.size());
    }

    template <Scalar T>
    State<T> Model<T>::make_state() const {
        return {
            .positions  = std::vector<T>(configuration_.anchors.size(), T{0}),
            .velocities = std::vector<T>(configuration_.anchors.size(), T{0}),
        };
    }

    template <Scalar T>
    StepCache<T> Model<T>::make_step_cache() const {
        return {.forces = {.values = std::vector<T>(configuration_.anchors.size(), T{0})}};
    }

    template <Scalar T>
    StateTangent<T> Model<T>::make_state_tangent() const {
        return {
            .positions  = std::vector<T>(configuration_.anchors.size(), T{0}),
            .velocities = std::vector<T>(configuration_.anchors.size(), T{0}),
        };
    }

    template <Scalar T>
    ControlTangent<T> Model<T>::make_control_tangent() const {
        return {.external_forces = std::vector<T>(configuration_.anchors.size(), T{0})};
    }

    template <Scalar T>
    ParameterTangent<T> Model<T>::make_parameter_tangent() const {
        return {
            .masses       = std::vector<T>(configuration_.anchors.size(), T{0}),
            .stiffnesses  = std::vector<T>(configuration_.edges.size(), T{0}),
            .dampings     = std::vector<T>(configuration_.edges.size(), T{0}),
            .rest_lengths = std::vector<T>(configuration_.edges.size(), T{0}),
        };
    }

    template <Scalar T>
    StateAdjoint<T> Model<T>::make_state_adjoint() const {
        return {
            .positions  = std::vector<T>(configuration_.anchors.size(), T{0}),
            .velocities = std::vector<T>(configuration_.anchors.size(), T{0}),
        };
    }

    template <Scalar T>
    ControlAdjoint<T> Model<T>::make_control_adjoint() const {
        return {.external_forces = std::vector<T>(configuration_.anchors.size(), T{0})};
    }

    template <Scalar T>
    ParameterAdjoint<T> Model<T>::make_parameter_adjoint() const {
        return {
            .masses       = std::vector<T>(configuration_.anchors.size(), T{0}),
            .stiffnesses  = std::vector<T>(configuration_.edges.size(), T{0}),
            .dampings     = std::vector<T>(configuration_.edges.size(), T{0}),
            .rest_lengths = std::vector<T>(configuration_.edges.size(), T{0}),
        };
    }

    template <Scalar T>
    void Model<T>::forward_step(const State<T>& state, const Control<T>& control, const Parameters<T>& parameters, State<T>& next_state, StepCache<T>& step_cache, ExecutionContext<T>& context) const {
        force_assembly_.forward(configuration_, state, control, parameters, step_cache.forces);
        semi_implicit_euler_.forward(configuration_, state, parameters, step_cache.forces, context.integrated_state);
        fixed_constraint_.forward(configuration_, context.integrated_state, next_state);
    }

    template <Scalar T>
    void Model<T>::jvp_step(const State<T>& state, const Control<T>&, const Parameters<T>& parameters, const State<T>&, const StepCache<T>& step_cache, const StateTangent<T>& state_tangent, const ControlTangent<T>& control_tangent, const ParameterTangent<T>& parameter_tangent, StateTangent<T>& next_state_tangent, ExecutionContext<T>& context) const {
        force_assembly_.jvp(configuration_, state, parameters, state_tangent, control_tangent, parameter_tangent, context.force_tangent);
        semi_implicit_euler_.jvp(configuration_, parameters, step_cache.forces, state_tangent, parameter_tangent, context.force_tangent, context.integrated_state_tangent);
        fixed_constraint_.jvp(configuration_, context.integrated_state_tangent, next_state_tangent);
    }

    template <Scalar T>
    void Model<T>::vjp_step(const State<T>& state, const Control<T>&, const Parameters<T>& parameters, const State<T>&, const StepCache<T>& step_cache, const StateAdjoint<T>& next_state_adjoint, StateAdjoint<T>& previous_state_adjoint, ControlAdjoint<T>& control_adjoint, ParameterAdjoint<T>& parameter_adjoint, ExecutionContext<T>& context) const {
        std::ranges::fill(context.force_adjoint.values, T{0});
        std::ranges::fill(context.integrated_state_adjoint.positions, T{0});
        std::ranges::fill(context.integrated_state_adjoint.velocities, T{0});
        fixed_constraint_.vjp(configuration_, next_state_adjoint, context.integrated_state_adjoint);
        semi_implicit_euler_.vjp(configuration_, parameters, step_cache.forces, context.integrated_state_adjoint, previous_state_adjoint, context.force_adjoint, parameter_adjoint);
        force_assembly_.vjp(configuration_, state, parameters, context.force_adjoint, previous_state_adjoint, control_adjoint, parameter_adjoint);
    }

    template class Model<float>;
    template class Model<double>;

} // namespace xayah::spring_chain
