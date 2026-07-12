export module xayah.spring_chain.data;

import std;

export namespace xayah::spring_chain {

    template <typename T>
    concept Scalar = std::same_as<T, float> || std::same_as<T, double>;

    struct Edge {
        std::size_t first;
        std::size_t second;
    };

    template <Scalar T>
    struct Configuration {
        std::vector<Edge> edges;
        std::vector<std::optional<T>> anchors;
        T time_step;
    };

    template <Scalar T>
    struct State {
        std::vector<T> positions;
        std::vector<T> velocities;
    };

    template <Scalar T>
    struct Control {
        std::vector<T> external_forces;
    };

    template <Scalar T>
    struct Parameters {
        std::vector<T> masses;
        std::vector<T> stiffnesses;
        std::vector<T> dampings;
        std::vector<T> rest_lengths;
    };

    template <Scalar T>
    struct Forces {
        std::vector<T> values;
    };

    template <Scalar T>
    struct StateTangent {
        std::vector<T> positions;
        std::vector<T> velocities;
    };

    template <Scalar T>
    struct ControlTangent {
        std::vector<T> external_forces;
    };

    template <Scalar T>
    struct ParameterTangent {
        std::vector<T> masses;
        std::vector<T> stiffnesses;
        std::vector<T> dampings;
        std::vector<T> rest_lengths;
    };

    template <Scalar T>
    struct ForceTangent {
        std::vector<T> values;
    };

    template <Scalar T>
    struct StateAdjoint {
        std::vector<T> positions;
        std::vector<T> velocities;

        StateAdjoint& operator+=(const StateAdjoint& other);
    };

    template <Scalar T>
    struct ControlAdjoint {
        std::vector<T> external_forces;
    };

    template <Scalar T>
    struct ParameterAdjoint {
        std::vector<T> masses;
        std::vector<T> stiffnesses;
        std::vector<T> dampings;
        std::vector<T> rest_lengths;
    };

    template <Scalar T>
    struct ForceAdjoint {
        std::vector<T> values;
    };

    template <Scalar T>
    struct StepCache {
        Forces<T> forces;
    };

    extern template struct StateAdjoint<float>;
    extern template struct StateAdjoint<double>;

} // namespace xayah::spring_chain
