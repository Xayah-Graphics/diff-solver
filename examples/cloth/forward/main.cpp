#include <cstdio>

import std;
import xayah.examples.cloth.forward;

int main() {
    try {
        xayah::cloth::examples::forward::ForwardSimulation simulation{};
        std::println("64 x 96 prescribed-traveling-load forward cloth");
        for (std::uint32_t step = 0u; step < 1440u; ++step) {
            simulation.step();
            if (simulation.metrics.step % 60u != 0u) continue;
            const xayah::cloth::examples::forward::ForwardSimulationMetrics& metrics = simulation.metrics;
            std::println("step {:4}  time {:.3f} s  load [{:+.4f}, {:+.4f}, {:+.4f}] m/s^2  free-edge displacement [{:+.5f}, {:+.5f}, {:+.5f}] m  velocity max {:.5f} m/s  kinetic {:.6f} J  strain stretch {:.5f} bending {:.5f}  step {:.3f} ms  average {:.3f} ms", metrics.step, metrics.physical_time, metrics.sampled_load_accelerations[0], metrics.sampled_load_accelerations[1], metrics.sampled_load_accelerations[2], metrics.free_edge_mean_displacement.x, metrics.free_edge_mean_displacement.y, metrics.free_edge_mean_displacement.z, metrics.maximum_velocity, metrics.kinetic_energy, metrics.maximum_absolute_stretch_strain, metrics.maximum_absolute_bending_strain, metrics.step_milliseconds, metrics.average_step_milliseconds);
        }
        return 0;
    } catch (const std::exception& error) {
        std::println(stderr, "diff-cloth-forward failed: {}", error.what());
        return 1;
    }
}
