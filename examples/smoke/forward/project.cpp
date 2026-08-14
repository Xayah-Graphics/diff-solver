module;

#include "project.h"

#include <cuda_runtime_api.h>

export module xayah.examples.smoke.forward.provider;

import spectra.sdk;
import spectra.sdk.cuda;
import std;
import xayah.examples.smoke.forward;
import xayah.smoke.data;

namespace {
    constexpr xayah::smoke::examples::forward::ForwardSimulationOptions simulation_options{.host_metrics = true};
    constexpr std::uint32_t emitter_count = 2u;
}

export namespace xayah::smoke::examples::forward::visualization {
    struct Settings {
        float density_scale{4.0F};
        bool show_emitters{true};
        float emitter_width{3.0F};
        float emitter_scale{0.04F};
    };

    struct Provider {
        Settings settings;

        static constexpr auto description = spectra::sdk::describe(
            "xayah.examples.smoke.forward",
            spectra::sdk::parameter<"density_scale", &Settings::density_scale>(
                "Density Scale",
                {},
                {.minimum = 0.05, .maximum = 20.0, .step = 0.05, .description = "Density multiplier applied directly to the published GPU grid.", .section = "Display"}
            ),
            spectra::sdk::parameter<"show_emitters", &Settings::show_emitters>(
                "Show Emitters",
                {},
                {.description = "Render the two source acceleration arrows.", .section = "Display"}
            ),
            spectra::sdk::parameter<"emitter_width", &Settings::emitter_width>(
                "Emitter Width",
                "px",
                {.minimum = 0.25, .maximum = 8.0, .step = 0.25, .description = "Screen-space emitter arrow width.", .section = "Display"}
            ),
            spectra::sdk::parameter<"emitter_scale", &Settings::emitter_scale>(
                "Emitter Scale",
                {},
                {.minimum = 0.01, .maximum = 0.12, .step = 0.005, .description = "World-space emitter arrow scale.", .section = "Display"}
            ),
            spectra::sdk::volume<"density">(
                spectra::sdk::field<"density", float>("Density"),
                spectra::sdk::field<"temperature", float>("Temperature", "K"),
                spectra::sdk::field<"velocity", spectra::sdk::MacFloat3>("Velocity", "m/s")
            ),
            spectra::sdk::vectors<"emitters">(),
            spectra::sdk::metric<"grid", spectra::sdk::Float3>("Grid", {}, "Simulation"),
            spectra::sdk::metric<"step", std::uint64_t>("Physical Step", {}, "Simulation"),
            spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true),
            spectra::sdk::metric<"pressure", std::uint32_t>("Pressure RBGS", "iterations", "Simulation"),
            spectra::sdk::metric<"vorticity", float>("Vorticity Confinement", {}, "Simulation"),
            spectra::sdk::metric<"density_mass", double>("Density Mass", {}, "Simulation", true),
            spectra::sdk::metric<"density_max", double>("Density Maximum", {}, "Simulation", true),
            spectra::sdk::metric<"temperature_max", double>("Temperature Maximum", {}, "Simulation", true),
            spectra::sdk::metric<"velocity_max", double>("Maximum Velocity", "m/s", "Simulation", true),
            spectra::sdk::metric<"cfl", double>("CFL", {}, "Simulation", true),
            spectra::sdk::metric<"pre_divergence", double>("Pre-Projection Divergence RMS", {}, "Simulation", true),
            spectra::sdk::metric<"post_divergence", double>("Post-Projection Divergence RMS", {}, "Simulation", true),
            spectra::sdk::metric<"divergence_ratio", double>("Divergence Ratio", {}, "Simulation", true)
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
        setup.volume<"density">({simulation_options.resolution[0], simulation_options.resolution[1], simulation_options.resolution[2]});
        setup.vectors<"emitters">(emitter_count);
    }

    void Provider::reset(const std::uint64_t) {
        simulation.emplace(simulation_options);
    }

    void Provider::step(const double) {
        simulation->step();
    }

    void Provider::publish(spectra::sdk::cuda::Output& output) {
        const cudaStream_t stream = static_cast<cudaStream_t>(simulation->context.resource->native_stream);
        spectra::sdk::cuda::Frame frame = output.begin(stream);
        spectra::sdk::cuda::Volume volume = frame.volume<"density">();
        std::span<float> density = volume.field<"density", float>();
        std::span<float> temperature = volume.field<"temperature", float>();
        spectra::sdk::cuda::MacField velocity = volume.field<"velocity">();
        project_cuda::launch_volume(
            stream,
            density.size(),
            simulation->current_state.density.values.data,
            simulation->current_state.temperature.values.data,
            settings.density_scale,
            density.data(),
            temperature.data()
        );
        if (cudaMemcpyAsync(velocity.x.data(), simulation->current_state.velocity.x.data, velocity.x.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Smoke velocity X publication failed");
        if (cudaMemcpyAsync(velocity.y.data(), simulation->current_state.velocity.y.data, velocity.y.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Smoke velocity Y publication failed");
        if (cudaMemcpyAsync(velocity.z.data(), simulation->current_state.velocity.z.data, velocity.z.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Smoke velocity Z publication failed");

        std::span<spectra::sdk::Vector> emitters = frame.vectors<"emitters">(settings.show_emitters ? emitter_count : 0u);
        if (settings.show_emitters) {
            const Vector3& left_origin = simulation->options.left_source_center;
            const Vector3& left_acceleration = simulation->options.left_acceleration;
            project_cuda::launch_emitter_vector(stream, left_origin.x, left_origin.y, left_origin.z, left_acceleration.x, left_acceleration.y, left_acceleration.z, settings.emitter_scale, settings.emitter_width, 0.10F, 0.84F, 0.96F, emitters.data());
            const Vector3& right_origin = simulation->options.right_source_center;
            const Vector3& right_acceleration = simulation->options.right_acceleration;
            project_cuda::launch_emitter_vector(stream, right_origin.x, right_origin.y, right_origin.z, right_acceleration.x, right_acceleration.y, right_acceleration.z, settings.emitter_scale, settings.emitter_width, 1.00F, 0.48F, 0.08F, emitters.data() + 1u);
        }

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error(std::format("smoke visualization kernel launch failed: {}", cudaGetErrorString(status)));
        const ForwardSimulationMetrics& metrics = simulation->metrics;
        frame.metric<"grid">().upload(spectra::sdk::Float3{static_cast<float>(simulation->options.resolution[0]), static_cast<float>(simulation->options.resolution[1]), static_cast<float>(simulation->options.resolution[2])});
        frame.metric<"step">().upload(metrics.step);
        frame.metric<"time">().upload(metrics.physical_time);
        frame.metric<"pressure">().upload(simulation->options.pressure_iterations);
        frame.metric<"vorticity">().upload(simulation->options.vorticity_confinement);
        frame.metric<"density_mass">().upload(metrics.density_mass);
        frame.metric<"density_max">().upload(metrics.density_maximum);
        frame.metric<"temperature_max">().upload(metrics.temperature_maximum);
        frame.metric<"velocity_max">().upload(metrics.maximum_velocity);
        frame.metric<"cfl">().upload(metrics.cfl);
        frame.metric<"pre_divergence">().upload(metrics.pre_projection_divergence_rms);
        frame.metric<"post_divergence">().upload(metrics.post_projection_divergence_rms);
        frame.metric<"divergence_ratio">().upload(metrics.divergence_ratio);
        frame.commit();
    }
}
