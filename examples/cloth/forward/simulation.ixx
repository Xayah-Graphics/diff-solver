export module xayah.examples.cloth.forward;

import std;
import xayah.cloth.data;
import xayah.cloth.model;

export namespace xayah::cloth::examples::forward {

    struct ForwardSimulationOptions {
        std::uint32_t rows{64u};
        std::uint32_t columns{96u};
        float width{3.0F};
        float height{2.0F};
        float time_step{1.0F / 240.0F};
        float mass{0.05F};
        float stretch_stiffness{400.0F};
        float stretch_damping{1.0F};
        float bending_stiffness{5.0F};
        float bending_damping{0.1F};
        float gravity_y{0.0F};
        float load_ramp_duration{0.5F};
        float load_period{1.2F};
        float load_base_acceleration{6.0F};
        float load_primary_acceleration{8.0F};
        float load_secondary_acceleration{2.0F};
    };

    struct ForwardSimulationMetrics {
        std::uint64_t step{};
        double physical_time{};
        double kinetic_energy{};
        double maximum_velocity{};
        double maximum_absolute_stretch_strain{};
        double maximum_absolute_bending_strain{};
        Vector3 free_edge_mean_position{};
        Vector3 free_edge_mean_displacement{};
        std::array<double, 3u> sampled_load_accelerations{};
        double step_milliseconds{};
        double average_step_milliseconds{};
    };

    struct ForwardSimulation {
        ForwardSimulationOptions options;
        Model model;
        ExecutionContext context;
        State current_state;
        ForwardSimulationMetrics metrics;

        explicit ForwardSimulation(ForwardSimulationOptions options = {});
        ~ForwardSimulation() noexcept;

        ForwardSimulation(const ForwardSimulation&) = delete;
        ForwardSimulation(ForwardSimulation&&) = delete;
        ForwardSimulation& operator=(const ForwardSimulation&) = delete;
        ForwardSimulation& operator=(ForwardSimulation&&) = delete;

        void reset();
        void step();

    private:
        State next_state_;
        Control control_;
        Parameters parameters_;
        StepCache step_cache_;
        double* device_metrics_;
    };

} // namespace xayah::cloth::examples::forward
