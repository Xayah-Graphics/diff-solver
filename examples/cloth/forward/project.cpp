module;

#include "project.h"

#include <cuda_runtime_api.h>

export module xayah.examples.cloth.forward.provider;

import spectra.sdk;
import spectra.sdk.cuda;
import std;
import xayah.cloth.data;
import xayah.examples.cloth.forward;

namespace {
    constexpr xayah::cloth::examples::forward::ForwardSimulationOptions simulation_options{};
    constexpr std::uint32_t vertex_count        = simulation_options.rows * simulation_options.columns;
    constexpr std::uint32_t stretch_count       = simulation_options.rows * (simulation_options.columns - 1u) + (simulation_options.rows - 1u) * simulation_options.columns + (simulation_options.rows - 1u) * (simulation_options.columns - 1u);
    constexpr std::uint32_t bending_count       = (simulation_options.rows - 2u) * (simulation_options.columns - 1u) + (simulation_options.rows - 1u) * (simulation_options.columns - 2u) + (simulation_options.rows - 1u) * (simulation_options.columns - 1u);
    constexpr std::uint32_t diagnostic_capacity = stretch_count + bending_count;
    constexpr std::uint32_t wind_capacity       = 3u;
}

export namespace xayah::cloth::examples::forward::visualization {
    struct Settings {
        float mass{0.0125F};
        float stretch_stiffness{600.0F};
        float stretch_damping{1.2F};
        float bending_stiffness{8.0F};
        float bending_damping{0.4F};
        float gravity_y{-0.1F};
        float wind_speed{6.0F};
        float gust_strength{0.35F};
        float gust_frequency{0.9F};
        float air_density{1.225F};
        float drag_coefficient{1.0F};
        float skin_drag_coefficient{0.10F};
        float wind_ramp_duration{0.5F};
        bool show_wind{true};
        bool show_stretch{};
        bool show_bending{};
        float wind_width{3.0F};
        float wind_scale{0.12F};
        float stretch_width{1.25F};
        float bending_width{1.0F};
        float strain_range{0.10F};
    };

    struct Provider {
        Settings settings;

        static constexpr auto description = spectra::sdk::describe(
            "xayah.examples.cloth.forward",
            spectra::sdk::parameter<"mass", &Settings::mass>("Particle Mass", "kg", {.minimum = 0.005, .maximum = 0.20, .step = 0.0025, .application = spectra::sdk::ParameterApplication::Reset, .description = "Mass assigned to every cloth particle.", .section = "Material"}),
            spectra::sdk::parameter<"stretch_stiffness", &Settings::stretch_stiffness>("Stretch Stiffness", "N/m", {.minimum = 25.0, .maximum = 1000.0, .step = 25.0, .application = spectra::sdk::ParameterApplication::Reset, .description = "In-plane stretch spring stiffness.", .section = "Material"}),
            spectra::sdk::parameter<"stretch_damping", &Settings::stretch_damping>("Stretch Damping", "N s/m", {.minimum = 0.0, .maximum = 10.0, .step = 0.1, .application = spectra::sdk::ParameterApplication::Reset, .description = "In-plane stretch spring damping.", .section = "Material"}),
            spectra::sdk::parameter<"bending_stiffness", &Settings::bending_stiffness>("Bending Stiffness", "N/m", {.minimum = 0.1, .maximum = 50.0, .step = 0.1, .application = spectra::sdk::ParameterApplication::Reset, .description = "Out-of-plane bending spring stiffness.", .section = "Material"}),
            spectra::sdk::parameter<"bending_damping", &Settings::bending_damping>("Bending Damping", "N s/m", {.minimum = 0.0, .maximum = 2.0, .step = 0.05, .application = spectra::sdk::ParameterApplication::Reset, .description = "Out-of-plane bending spring damping.", .section = "Material"}),
            spectra::sdk::parameter<"gravity_y", &Settings::gravity_y>("Gravity", "m/s^2", {.minimum = -9.81, .maximum = 0.0, .step = 0.05, .application = spectra::sdk::ParameterApplication::Reset, .description = "Vertical acceleration applied to the flag.", .section = "Simulation"}),
            spectra::sdk::parameter<"wind_speed", &Settings::wind_speed>("Wind Speed", "m/s", {.minimum = 0.0, .maximum = 20.0, .step = 0.25, .application = spectra::sdk::ParameterApplication::Reset, .description = "Mean wind speed from the pole toward the free edge.", .section = "Wind"}),
            spectra::sdk::parameter<"gust_strength", &Settings::gust_strength>("Gust Strength", {}, {.minimum = 0.0, .maximum = 0.50, .step = 0.01, .application = spectra::sdk::ParameterApplication::Reset, .description = "Fractional amplitude of the spatially varying wind field.", .section = "Wind"}),
            spectra::sdk::parameter<"gust_frequency", &Settings::gust_frequency>("Gust Frequency", "Hz", {.minimum = 0.10, .maximum = 3.0, .step = 0.05, .application = spectra::sdk::ParameterApplication::Reset, .description = "Base temporal frequency of the wind variation.", .section = "Wind"}),
            spectra::sdk::parameter<"air_density", &Settings::air_density>("Air Density", "kg/m^3", {.minimum = 0.10, .maximum = 2.0, .step = 0.025, .application = spectra::sdk::ParameterApplication::Reset, .description = "Density used by the aerodynamic pressure model.", .section = "Wind"}),
            spectra::sdk::parameter<"drag_coefficient", &Settings::drag_coefficient>("Normal Drag", {}, {.minimum = 0.10, .maximum = 3.0, .step = 0.05, .application = spectra::sdk::ParameterApplication::Reset, .description = "Double-sided pressure drag normal to the fabric.", .section = "Wind"}),
            spectra::sdk::parameter<"skin_drag_coefficient", &Settings::skin_drag_coefficient>("Skin Drag", {}, {.minimum = 0.0, .maximum = 0.50, .step = 0.01, .application = spectra::sdk::ParameterApplication::Reset, .description = "Tangential surface drag that keeps the flag extended along the flow.", .section = "Wind"}),
            spectra::sdk::parameter<"wind_ramp_duration", &Settings::wind_ramp_duration>("Wind Ramp", "s", {.minimum = 0.10, .maximum = 3.0, .step = 0.05, .application = spectra::sdk::ParameterApplication::Reset, .description = "Smooth startup duration of the airflow.", .section = "Wind"}),
            spectra::sdk::parameter<"show_wind", &Settings::show_wind>("Show Wind", {}, {.description = "Render three samples of the wind field.", .section = "Display"}),
            spectra::sdk::parameter<"show_stretch", &Settings::show_stretch>("Show Stretch Strain", {}, {.description = "Render stretch springs with strain coloring.", .section = "Display"}),
            spectra::sdk::parameter<"show_bending", &Settings::show_bending>("Show Bending", {}, {.description = "Render bending springs over the cloth surface.", .section = "Display"}),
            spectra::sdk::parameter<"wind_width", &Settings::wind_width>("Wind Width", "px", {.minimum = 0.25, .maximum = 8.0, .step = 0.25, .description = "Screen-space width of wind arrows.", .section = "Display"}),
            spectra::sdk::parameter<"wind_scale", &Settings::wind_scale>("Wind Scale", "s", {.minimum = 0.01, .maximum = 0.30, .step = 0.01, .description = "World-space scale applied to wind velocity arrows.", .section = "Display"}),
            spectra::sdk::parameter<"stretch_width", &Settings::stretch_width>("Stretch Width", "px", {.minimum = 0.25, .maximum = 8.0, .step = 0.25, .description = "Screen-space width of stretch springs.", .section = "Display"}),
            spectra::sdk::parameter<"bending_width", &Settings::bending_width>("Bending Width", "px", {.minimum = 0.25, .maximum = 8.0, .step = 0.25, .description = "Screen-space width of bending springs.", .section = "Display"}),
            spectra::sdk::parameter<"strain_range", &Settings::strain_range>("Strain Range", {}, {.minimum = 0.01, .maximum = 0.50, .step = 0.01, .description = "Absolute strain mapped to the visualization color extremes.", .section = "Display"}),
            spectra::sdk::mesh<"surface">(),
            spectra::sdk::lines<"diagnostics">(),
            spectra::sdk::vectors<"wind">(),
            spectra::sdk::metric<"grid", spectra::sdk::Float3>("Grid", {}, "Simulation"),
            spectra::sdk::metric<"step", std::uint64_t>("Physical Step", {}, "Simulation"),
            spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true),
            spectra::sdk::metric<"free_edge_position", spectra::sdk::Float3>("Free Edge Mean Position", "m", "Simulation"),
            spectra::sdk::metric<"free_edge_displacement", spectra::sdk::Float3>("Free Edge Mean Displacement", "m", "Simulation"),
            spectra::sdk::metric<"wind_quarter", spectra::sdk::Float3>("Wind u=0.25", "m/s", "Wind"),
            spectra::sdk::metric<"wind_half", spectra::sdk::Float3>("Wind u=0.50", "m/s", "Wind"),
            spectra::sdk::metric<"wind_three_quarters", spectra::sdk::Float3>("Wind u=0.75", "m/s", "Wind"),
            spectra::sdk::metric<"aerodynamic_force", double>("Aerodynamic Force", "N", "Wind", true),
            spectra::sdk::metric<"velocity", double>("Maximum Velocity", "m/s", "Dynamics", true),
            spectra::sdk::metric<"kinetic", double>("Kinetic Energy", "J", "Dynamics", true),
            spectra::sdk::metric<"stretch_strain", double>("Maximum Stretch Strain", {}, "Dynamics", true),
            spectra::sdk::metric<"bending_strain", double>("Maximum Bending Strain", {}, "Dynamics", true),
            spectra::sdk::metric<"step_time", double>("Last Step", "ms", "Performance", true),
            spectra::sdk::metric<"average_step_time", double>("Average Step", "ms", "Performance", true),
            spectra::sdk::metric<"stretch_material", spectra::sdk::Float3>("Stretch Stiffness / Damping", {}, "Material"),
            spectra::sdk::metric<"bending_material", spectra::sdk::Float3>("Bending Stiffness / Damping", {}, "Material")
        );

        Provider(Settings settings, const std::filesystem::path& assets);

        void setup(spectra::sdk::cuda::Setup& setup);
        void reset(std::uint64_t seed);
        void step(double seconds);
        void publish(spectra::sdk::cuda::Output& output);

    private:
        std::optional<ForwardSimulation> simulation;
    };

    Provider::Provider(Settings source, const std::filesystem::path&) : settings(source) {}

    void Provider::setup(spectra::sdk::cuda::Setup& setup) {
        static_cast<void>(setup.mesh<"surface">(vertex_count, 0u));
        setup.lines<"diagnostics">(diagnostic_capacity);
        setup.vectors<"wind">(wind_capacity);
    }

    void Provider::reset(const std::uint64_t) {
        ForwardSimulationOptions options = simulation_options;
        options.mass                  = settings.mass;
        options.stretch_stiffness     = settings.stretch_stiffness;
        options.stretch_damping       = settings.stretch_damping;
        options.bending_stiffness     = settings.bending_stiffness;
        options.bending_damping       = settings.bending_damping;
        options.gravity_y             = settings.gravity_y;
        options.wind_speed            = settings.wind_speed;
        options.gust_strength         = settings.gust_strength;
        options.gust_frequency        = settings.gust_frequency;
        options.air_density           = settings.air_density;
        options.drag_coefficient      = settings.drag_coefficient;
        options.skin_drag_coefficient = settings.skin_drag_coefficient;
        options.wind_ramp_duration    = settings.wind_ramp_duration;
        simulation.emplace(options);
    }

    void Provider::step(const double) {
        simulation->step();
    }

    void Provider::publish(spectra::sdk::cuda::Output& output) {
        const cudaStream_t stream = static_cast<cudaStream_t>(simulation->context.resource->native_stream);
        spectra::sdk::cuda::Frame frame = output.begin(stream);
        spectra::sdk::cuda::Mesh surface = frame.mesh<"surface">();
        const State& state = simulation->current_state;
        project_cuda::launch_positions(stream, vertex_count, state.positions.x.data, state.positions.y.data, state.positions.z.data, surface.positions.data());
        surface.vertex_count   = vertex_count;
        surface.triangle_count = 0u;

        const std::uint32_t diagnostic_count = (settings.show_stretch ? stretch_count : 0u) + (settings.show_bending ? bending_count : 0u);
        std::span<spectra::sdk::Line> diagnostics = frame.lines<"diagnostics">(diagnostic_count);
        const DeviceTopology& topology = simulation->context.device_topology;
        std::uint32_t diagnostic_offset{};
        if (settings.show_stretch) {
            project_cuda::launch_segments(stream, stretch_count, state.positions.x.data, state.positions.y.data, state.positions.z.data, topology.stretch.first.data, topology.stretch.second.data, simulation->parameters.stretch_rest_lengths.data, settings.stretch_width, settings.strain_range, project_cuda::SegmentStyle::Stretch, diagnostics.data());
            diagnostic_offset += stretch_count;
        }
        if (settings.show_bending) {
            project_cuda::launch_segments(stream, bending_count, state.positions.x.data, state.positions.y.data, state.positions.z.data, topology.bending.first.data, topology.bending.second.data, simulation->parameters.bending_rest_lengths.data, settings.bending_width, settings.strain_range, project_cuda::SegmentStyle::Bending, diagnostics.data() + diagnostic_offset);
        }
        std::span<spectra::sdk::Vector> wind_vectors = frame.vectors<"wind">(settings.show_wind ? wind_capacity : 0u);
        if (settings.show_wind)
            for (std::uint32_t sample = 0u; sample != wind_capacity; ++sample) {
                const Vector3& wind = simulation->metrics.sampled_wind_velocities[sample];
                project_cuda::launch_wind_vector(stream, -1.05F, -0.35F - 0.65F * static_cast<float>(sample), 0.15F, wind.x, wind.z, settings.wind_scale, settings.wind_width, wind_vectors.data() + sample);
            }
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error(std::format("cloth visualization kernel launch failed: {}", cudaGetErrorString(status)));

        const ForwardSimulationMetrics& metrics = simulation->metrics;
        frame.metric<"grid">().upload(spectra::sdk::Float3{static_cast<float>(simulation->options.columns), static_cast<float>(simulation->options.rows), 1.0F});
        frame.metric<"step">().upload(metrics.step);
        frame.metric<"time">().upload(metrics.physical_time);
        frame.metric<"free_edge_position">().upload(spectra::sdk::Float3{metrics.free_edge_mean_position.x, metrics.free_edge_mean_position.y, metrics.free_edge_mean_position.z});
        frame.metric<"free_edge_displacement">().upload(spectra::sdk::Float3{metrics.free_edge_mean_displacement.x, metrics.free_edge_mean_displacement.y, metrics.free_edge_mean_displacement.z});
        frame.metric<"wind_quarter">().upload(spectra::sdk::Float3{metrics.sampled_wind_velocities[0].x, metrics.sampled_wind_velocities[0].y, metrics.sampled_wind_velocities[0].z});
        frame.metric<"wind_half">().upload(spectra::sdk::Float3{metrics.sampled_wind_velocities[1].x, metrics.sampled_wind_velocities[1].y, metrics.sampled_wind_velocities[1].z});
        frame.metric<"wind_three_quarters">().upload(spectra::sdk::Float3{metrics.sampled_wind_velocities[2].x, metrics.sampled_wind_velocities[2].y, metrics.sampled_wind_velocities[2].z});
        frame.metric<"aerodynamic_force">().upload(metrics.aerodynamic_force);
        frame.metric<"velocity">().upload(metrics.maximum_velocity);
        frame.metric<"kinetic">().upload(metrics.kinetic_energy);
        frame.metric<"stretch_strain">().upload(metrics.maximum_absolute_stretch_strain);
        frame.metric<"bending_strain">().upload(metrics.maximum_absolute_bending_strain);
        frame.metric<"step_time">().upload(metrics.step_milliseconds);
        frame.metric<"average_step_time">().upload(metrics.average_step_milliseconds);
        frame.metric<"stretch_material">().upload(spectra::sdk::Float3{simulation->options.stretch_stiffness, simulation->options.stretch_damping, 0.0F});
        frame.metric<"bending_material">().upload(spectra::sdk::Float3{simulation->options.bending_stiffness, simulation->options.bending_damping, 0.0F});
        frame.commit();
    }
}
