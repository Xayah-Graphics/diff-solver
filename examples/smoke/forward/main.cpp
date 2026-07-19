#include <cstdio>
#include <cuda_runtime_api.h>

import std;
import xayah.examples.smoke.forward;

int main() {
    try {
        std::size_t free_memory_before{};
        std::size_t total_memory{};
        if (const cudaError_t status = cudaMemGetInfo(&free_memory_before, &total_memory); status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        xayah::smoke::examples::forward::ForwardSimulation simulation{};
        std::size_t free_memory_after{};
        if (const cudaError_t status = cudaMemGetInfo(&free_memory_after, &total_memory); status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        std::println("128 x 192 x 128 double-jet forward smoke");
        std::println("GPU allocation: {:.1f} MiB", static_cast<double>(free_memory_before - free_memory_after) / (1024.0 * 1024.0));
        for (std::uint32_t step = 0u; step < 480u; ++step) {
            simulation.step();
            if (simulation.metrics.step % 30u != 0u) continue;
            const xayah::smoke::examples::forward::ForwardSimulationMetrics& metrics = simulation.metrics;
            std::println(
                "step {:3}  time {:.3f} s  density mass {:.6f} max {:.5f}  temperature max {:.5f}  velocity max {:.5f}  CFL {:.4f}  divergence {:.4e} -> {:.4e} ({:.4f})  step {:.3f} ms  average {:.3f} ms",
                metrics.step,
                metrics.physical_time,
                metrics.density_mass,
                metrics.density_maximum,
                metrics.temperature_maximum,
                metrics.maximum_velocity,
                metrics.cfl,
                metrics.pre_projection_divergence_rms,
                metrics.post_projection_divergence_rms,
                metrics.divergence_ratio,
                metrics.step_milliseconds,
                metrics.average_step_milliseconds);
        }
        return 0;
    } catch (const std::exception& error) {
        std::println(stderr, "diff-smoke-forward failed: {}", error.what());
        return 1;
    }
}
