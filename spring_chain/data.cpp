module xayah.spring_chain.data;

import std;

namespace xayah::spring_chain {

    template <Scalar T>
    StateAdjoint<T>& StateAdjoint<T>::operator+=(const StateAdjoint& other) {
        for (std::size_t particle = 0; particle < positions.size(); ++particle) positions[particle] += other.positions[particle];
        for (std::size_t particle = 0; particle < velocities.size(); ++particle) velocities[particle] += other.velocities[particle];
        return *this;
    }

    template struct StateAdjoint<float>;
    template struct StateAdjoint<double>;

} // namespace xayah::spring_chain
