export module xayah.examples.smoke.forward;

import std;
import xayah.smoke.data;
import xayah.smoke.model;

export namespace xayah::smoke::examples::forward {

    struct ForwardSimulationOptions {
        std::array<std::uint32_t, 3u> resolution{128u, 192u, 128u};
        float cell_size{1.0F / 128.0F};
        float time_step{1.0F / 120.0F};
        std::uint32_t pressure_iterations{160u};
        float density_source_rate{4.0F};
        float temperature_source_rate{8.0F};
        float ambient_temperature{};
        float density_buoyancy{-0.1F};
        float temperature_buoyancy{1.0F};
        float vorticity_confinement{2.0F};
        Vector3 left_source_center{0.25F, 0.10F, 0.36F};
        Vector3 right_source_center{0.75F, 0.10F, 0.64F};
        float source_radius{0.055F};
        Vector3 left_acceleration{3.5F, 5.0F, 1.8F};
        Vector3 right_acceleration{-3.5F, 5.0F, -1.8F};
        float pulse_period{0.9F};
    };

    struct ForwardSimulationMetrics {
        std::uint64_t step{};
        double physical_time{};
        double density_mass{};
        double density_maximum{};
        double temperature_maximum{};
        double maximum_velocity{};
        double cfl{};
        double pre_projection_divergence_rms{};
        double post_projection_divergence_rms{};
        double divergence_ratio{};
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

        ForwardSimulation(const ForwardSimulation&)            = delete;
        ForwardSimulation(ForwardSimulation&&)                 = delete;
        ForwardSimulation& operator=(const ForwardSimulation&) = delete;
        ForwardSimulation& operator=(ForwardSimulation&&)      = delete;

        void reset();
        void step();

    private:
        State next_state_;
        Control control_;
        Parameters parameters_;
        StepCache step_cache_;
        double* device_metrics_;
    };

} // namespace xayah::smoke::examples::forward
