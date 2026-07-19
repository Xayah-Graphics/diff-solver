#if defined(_WIN32)
#define XAYAH_SMOKE_WIND_TRAJECTORY_OPTIMIZATION_PLUGIN_EXPORT __declspec(dllexport)
#else
#define XAYAH_SMOKE_WIND_TRAJECTORY_OPTIMIZATION_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime_api.h>

#include "project.h"

import std;
import xayah.examples.smoke.wind_trajectory_optimization;
import xayah.smoke.data;
import xayah.smoke.model;
import xayah.spectra.plugin;

namespace xayah::smoke::examples::wind_trajectory_optimization::project {

    namespace plugin = spectra::plugin;

    namespace {

        constexpr std::uint64_t segment_bytes = 48u;
        constexpr std::uint32_t trajectory_steps = 90u;

        struct ExternalBuffer {
            ExternalBuffer() = default;

            ~ExternalBuffer() noexcept {
                for (void* const mapped_buffer : mapped_buffers) if (mapped_buffer != nullptr && cudaFree(mapped_buffer) != cudaSuccess) std::terminate();
                for (const cudaExternalMemory_t external_memory : external_memories_) if (external_memory != nullptr && cudaDestroyExternalMemory(external_memory) != cudaSuccess) std::terminate();
                for (const cudaEvent_t ready_event : ready_events) if (ready_event != nullptr && cudaEventDestroy(ready_event) != cudaSuccess) std::terminate();
                if (allocation.resource_id != 0u) {
                    try {
                        host_services_->release_gpu_buffer(allocation.resource_id);
                    } catch (...) {
                        std::terminate();
                    }
                }
            }

            ExternalBuffer(const ExternalBuffer&) = delete;
            ExternalBuffer(ExternalBuffer&&) = delete;
            ExternalBuffer& operator=(const ExternalBuffer&) = delete;
            ExternalBuffer& operator=(ExternalBuffer&&) = delete;

            void create(std::shared_ptr<plugin::HostServices> host_services, const std::uint32_t kind, const std::uint64_t byte_size) {
                plugin::GpuBufferAllocation next_allocation = host_services->request_gpu_buffer(kind, byte_size);
                std::vector<cudaExternalMemory_t> external_memories{};
                std::vector<void*> next_mapped_buffers{};
                std::vector<cudaEvent_t> next_ready_events{};
                external_memories.reserve(next_allocation.slots.size());
                next_mapped_buffers.reserve(next_allocation.slots.size());
                next_ready_events.reserve(next_allocation.slots.size());
                try {
                    for (plugin::GpuBufferSlotAllocation& slot : next_allocation.slots) {
                        cudaExternalMemoryHandleDesc memory_descriptor{};
                        memory_descriptor.size = next_allocation.byte_size;
#if defined(_WIN32)
                        memory_descriptor.type = cudaExternalMemoryHandleTypeOpaqueWin32;
                        memory_descriptor.handle.win32.handle = reinterpret_cast<void*>(slot.handle);
#else
                        memory_descriptor.type = cudaExternalMemoryHandleTypeOpaqueFd;
                        memory_descriptor.handle.fd = static_cast<int>(slot.handle);
#endif
                        cudaExternalMemory_t external_memory{};
                        const cudaError_t import_status = cudaImportExternalMemory(&external_memory, &memory_descriptor);
#if defined(_WIN32)
                        close_imported_handle(slot);
#else
                        if (import_status == cudaSuccess) slot.handle = 0u;
                        else close_imported_handle(slot);
#endif
                        if (import_status != cudaSuccess) throw std::runtime_error(std::format("cudaImportExternalMemory failed: {}", cudaGetErrorString(import_status)));
                        cudaExternalMemoryBufferDesc buffer_descriptor{};
                        buffer_descriptor.size = next_allocation.byte_size;
                        void* mapped_buffer{};
                        if (const cudaError_t status = cudaExternalMemoryGetMappedBuffer(&mapped_buffer, external_memory, &buffer_descriptor); status != cudaSuccess) {
                            static_cast<void>(cudaDestroyExternalMemory(external_memory));
                            throw std::runtime_error(std::format("cudaExternalMemoryGetMappedBuffer failed: {}", cudaGetErrorString(status)));
                        }
                        cudaEvent_t ready_event{};
                        if (const cudaError_t status = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming); status != cudaSuccess) {
                            static_cast<void>(cudaFree(mapped_buffer));
                            static_cast<void>(cudaDestroyExternalMemory(external_memory));
                            throw std::runtime_error(std::format("cudaEventCreateWithFlags failed: {}", cudaGetErrorString(status)));
                        }
                        external_memories.push_back(external_memory);
                        next_mapped_buffers.push_back(mapped_buffer);
                        next_ready_events.push_back(ready_event);
                    }
                } catch (...) {
                    for (plugin::GpuBufferSlotAllocation& slot : next_allocation.slots) if (slot.handle != 0u) close_imported_handle(slot);
                    for (void* const mapped_buffer : next_mapped_buffers) if (mapped_buffer != nullptr) static_cast<void>(cudaFree(mapped_buffer));
                    for (const cudaExternalMemory_t external_memory : external_memories) if (external_memory != nullptr) static_cast<void>(cudaDestroyExternalMemory(external_memory));
                    for (const cudaEvent_t ready_event : next_ready_events) if (ready_event != nullptr) static_cast<void>(cudaEventDestroy(ready_event));
                    host_services->release_gpu_buffer(next_allocation.resource_id);
                    throw;
                }
                host_services_ = std::move(host_services);
                allocation = std::move(next_allocation);
                external_memories_ = std::move(external_memories);
                mapped_buffers = std::move(next_mapped_buffers);
                ready_events = std::move(next_ready_events);
                slot_revisions.assign(mapped_buffers.size(), 0u);
            }

            void record_ready(const std::uint32_t frame_slot_index, const cudaStream_t stream) {
                if (const cudaError_t status = cudaEventRecord(ready_events[frame_slot_index], stream); status != cudaSuccess) throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorString(status)));
            }

            plugin::GpuBufferAllocation allocation{};
            std::vector<void*> mapped_buffers{};
            std::vector<cudaEvent_t> ready_events{};
            std::vector<std::uint64_t> slot_revisions{};

        private:
            static void close_imported_handle(plugin::GpuBufferSlotAllocation& slot) noexcept {
#if defined(_WIN32)
                if (slot.handle_kind == plugin::GpuResourceHandleKind::OpaqueWin32 && slot.handle != 0u && CloseHandle(reinterpret_cast<HANDLE>(slot.handle)) == 0) std::terminate();
#else
                if (slot.handle_kind == plugin::GpuResourceHandleKind::OpaqueFileDescriptor && slot.handle != 0u && close(static_cast<int>(slot.handle)) != 0) std::terminate();
#endif
                slot.handle = 0u;
            }

            std::shared_ptr<plugin::HostServices> host_services_{};
            std::vector<cudaExternalMemory_t> external_memories_{};
        };

        struct ProjectOptions {
            float density_source_rate{3.0F};
            float temperature_source_rate{6.0F};
            float density_buoyancy{-0.1F};
            float temperature_buoyancy{1.0F};
            float adam_learning_rate{0.05F};
            std::uint32_t iterations_per_update{1u};
        };

        plugin::Camera overview_camera() {
            const std::array<float, 3u> position{1.5F, 1.05F, 4.8F};
            const std::array<float, 3u> target{1.5F, 0.72F, 0.50F};
            std::array<float, 3u> forward{target[0] - position[0], target[1] - position[1], target[2] - position[2]};
            const float forward_length = std::sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
            for (float& component : forward) component /= forward_length;
            std::array<float, 3u> right{-forward[2], 0.0F, forward[0]};
            const float right_length = std::sqrt(right[0] * right[0] + right[2] * right[2]);
            for (float& component : right) component /= right_length;
            const std::array<float, 3u> down{
                forward[1] * right[2] - forward[2] * right[1],
                forward[2] * right[0] - forward[0] * right[2],
                forward[0] * right[1] - forward[1] * right[0],
            };
            return {
                .name = "Overview",
                .position = position,
                .right = right,
                .down = down,
                .forward = forward,
                .vertical_fov_degrees = 38.0F,
                .near_plane = 0.01F,
                .far_plane = 20.0F,
            };
        }

        Vector3 interpolate_wind(const std::span<const Vector3> keyframes, const std::size_t control_step, const std::size_t trajectory_steps) {
            const double coordinate = static_cast<double>(control_step) * static_cast<double>(keyframes.size() - 1u) / static_cast<double>(trajectory_steps - 1u);
            const std::size_t first = (std::min)(static_cast<std::size_t>(coordinate), keyframes.size() - 2u);
            const std::size_t second = first + 1u;
            const float weight = static_cast<float>(coordinate - static_cast<double>(first));
            return {
                .x = (1.0F - weight) * keyframes[first].x + weight * keyframes[second].x,
                .y = 0.0F,
                .z = (1.0F - weight) * keyframes[first].z + weight * keyframes[second].z,
            };
        }

    } // namespace

    struct Project final {
        std::uint64_t revision{1u};

        Project(ProjectOptions options, std::shared_ptr<plugin::HostServices> host_services);
        Project(const Project&) = delete;
        Project(Project&&) = delete;
        Project& operator=(const Project&) = delete;
        Project& operator=(Project&&) = delete;

        [[nodiscard]] static const plugin::PluginDefinition<Project>& plugin();
        [[nodiscard]] static Project open(plugin::OpenContext context);

        void update(const plugin::UpdateInfo& update);
        void write_document(plugin::SceneBuilder& scene) const;
        void write_frame(plugin::SceneBuilder& scene, plugin::FrameInfo frame) const;
        void write_controls(plugin::ControlBuilder& controls) const;

    private:
        void write_visualization(std::uint32_t frame_slot_index);

        ProjectOptions options_{};
        WindTrajectoryOptimizationTask task_;
        ExternalBuffer density_{};
        ExternalBuffer color_{};
        ExternalBuffer target_wind_{};
        ExternalBuffer estimated_wind_{};
        std::uint64_t density_value_bytes_{};
        std::uint64_t content_revision_{1u};
        float optimization_milliseconds_{};
        std::uint32_t current_frame_slot_{};
        std::uint32_t trajectory_step_;
        double trajectory_playback_accumulator_{};
        bool update_running_{};
        bool reset_pending_{};
        bool trajectory_playback_{};
        bool loop_trajectory_{true};
        bool show_target_{true};
        bool show_estimate_{true};
        bool show_difference_{true};
        bool show_wind_{true};
        float trajectory_fps_{15.0F};
        float smoke_opacity_scale_{3.0F};
        float difference_opacity_scale_{20.0F};
        float wind_width_{3.0F};
        float wind_scale_{0.18F};
        double target_density_mean_{};
        double target_density_maximum_{};
        double estimated_density_mean_{};
        double estimated_density_maximum_{};
    };

    Project::Project(ProjectOptions options, std::shared_ptr<plugin::HostServices> host_services)
        : options_(options), task_(WindTrajectoryOptimizationOptions{
              .trajectory_steps = trajectory_steps,
              .density_source_rate = options_.density_source_rate,
              .temperature_source_rate = options_.temperature_source_rate,
              .ambient_temperature = 0.0F,
              .density_buoyancy = options_.density_buoyancy,
              .temperature_buoyancy = options_.temperature_buoyancy,
              .adam_learning_rate = options_.adam_learning_rate,
          }), trajectory_step_(54u) {
        const std::uint64_t output_cells = 3u * static_cast<std::uint64_t>(task_.model.configuration.resolution[0]) * task_.model.configuration.resolution[1] * task_.model.configuration.resolution[2];
        density_value_bytes_ = output_cells * sizeof(float);
        density_.create(host_services, plugin::GpuBufferKindVolumeChannel, density_value_bytes_ + 4u * sizeof(double));
        color_.create(host_services, plugin::GpuBufferKindVolumeChannel, 3u * output_cells * sizeof(float));
        target_wind_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, 3u * segment_bytes);
        estimated_wind_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, 3u * segment_bytes);
        for (std::size_t frame_slot = 0u; frame_slot < density_.mapped_buffers.size(); ++frame_slot) write_visualization(static_cast<std::uint32_t>(frame_slot));
        task_.context.synchronize();
    }

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition = [] {
            plugin::PluginDefinition<Project> value{
                .id = "xayah.examples.smoke.wind_trajectory_optimization",
                .title = "CUDA Smoke Wind Trajectory Optimization",
                .open_action_label = "Open CUDA Smoke Wind Trajectory Optimization",
                .sections = {{.id = "optimization", .label = "Optimization"}, {.id = "physics", .label = "Physics"}, {.id = "display", .label = "Display"}},
                .open_options = {
                    plugin::float_value("density_source_rate", "Density Source Rate", 3.0F, "physics"),
                    plugin::float_value("temperature_source_rate", "Temperature Source Rate", 6.0F, "physics"),
                    plugin::float_value("density_buoyancy", "Density Buoyancy", -0.1F, "physics"),
                    plugin::float_value("temperature_buoyancy", "Temperature Buoyancy", 1.0F, "physics"),
                    plugin::float_value("adam_learning_rate", "Adam Learning Rate", 0.05F, "optimization"),
                    plugin::unsigned_integer("iterations_per_update", "Iterations Per Update", 1u, "optimization"),
                },
                .actions = {plugin::action<Project>("reset", "Reset Optimization", "Restore zero wind keyframes and Adam state on the next safe frame-slot update.", "optimization", [](Project& project) { project.reset_pending_ = true; })},
                .settings = {
                    plugin::unsigned_integer_setting<Project>("trajectory_frame", "Trajectory Frame", 54u, "display", 0u, 90u, 1u, [](Project& project, const std::uint64_t value) {
                        project.trajectory_step_ = static_cast<std::uint32_t>(value);
                        project.trajectory_playback_ = false;
                        project.trajectory_playback_accumulator_ = 0.0;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("play_trajectory", "Play Trajectory", false, "display", [](Project& project, const bool value) {
                        project.trajectory_playback_ = value;
                        project.trajectory_playback_accumulator_ = 0.0;
                    }),
                    plugin::toggle<Project>("loop_trajectory", "Loop Trajectory", true, "display", [](Project& project, const bool value) {
                        project.loop_trajectory_ = value;
                    }),
                    plugin::toggle<Project>("show_target", "Show Target", true, "display", [](Project& project, const bool value) {
                        project.show_target_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_estimate", "Show Estimate", true, "display", [](Project& project, const bool value) {
                        project.show_estimate_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_difference", "Show Difference", true, "display", [](Project& project, const bool value) {
                        project.show_difference_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_wind", "Show Wind", true, "display", [](Project& project, const bool value) {
                        project.show_wind_ = value;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("trajectory_fps", "Trajectory FPS", 15.0F, "display", 1.0F, 60.0F, 1.0F, [](Project& project, const float value) {
                        project.trajectory_fps_ = value;
                    }),
                    plugin::float_setting<Project>("smoke_opacity_scale", "Smoke Opacity Scale", 3.0F, "display", 0.05F, 20.0F, 0.05F, [](Project& project, const float value) {
                        project.smoke_opacity_scale_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("difference_opacity_scale", "Difference Opacity Scale", 20.0F, "display", 0.1F, 50.0F, 0.1F, [](Project& project, const float value) {
                        project.difference_opacity_scale_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("wind_width", "Wind Width", 3.0F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.wind_width_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("wind_scale", "Wind Scale", 0.18F, "display", 0.02F, 1.0F, 0.02F, [](Project& project, const float value) {
                        project.wind_scale_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                },
            };
            return value;
        }();
        return definition;
    }

    Project Project::open(plugin::OpenContext context) {
        ProjectOptions options{};
        for (const plugin::Option& option : context.options) {
            if (option.key == "density_source_rate") options.density_source_rate = std::stof(option.value);
            else if (option.key == "temperature_source_rate") options.temperature_source_rate = std::stof(option.value);
            else if (option.key == "density_buoyancy") options.density_buoyancy = std::stof(option.value);
            else if (option.key == "temperature_buoyancy") options.temperature_buoyancy = std::stof(option.value);
            else if (option.key == "adam_learning_rate") options.adam_learning_rate = std::stof(option.value);
            else if (option.key == "iterations_per_update") options.iterations_per_update = static_cast<std::uint32_t>(std::stoul(option.value));
        }
        return Project(options, std::move(context.host_services));
    }

    void Project::update(const plugin::UpdateInfo& update) {
        current_frame_slot_ = update.frame_slot_index;
        update_running_ = update.update_running;
        if (reset_pending_) {
            task_.reset();
            reset_pending_ = false;
            ++content_revision_;
            ++revision;
        }
        if (update.update_delta_seconds > 0.0) {
            const auto begin = std::chrono::steady_clock::now();
            for (std::uint32_t iteration = 0u; iteration < options_.iterations_per_update; ++iteration) task_.optimize_step();
            optimization_milliseconds_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
            ++content_revision_;
            ++revision;
        }
        if (trajectory_playback_) {
            trajectory_playback_accumulator_ += update.wall_delta_seconds * trajectory_fps_;
            const std::uint32_t frame_advance = static_cast<std::uint32_t>(trajectory_playback_accumulator_);
            if (frame_advance > 0u) {
                trajectory_playback_accumulator_ -= frame_advance;
                if (loop_trajectory_) trajectory_step_ = (trajectory_step_ + frame_advance) % (trajectory_steps + 1u);
                else {
                    trajectory_step_ = std::min(trajectory_step_ + frame_advance, trajectory_steps);
                    if (trajectory_step_ == trajectory_steps) {
                        trajectory_playback_ = false;
                        trajectory_playback_accumulator_ = 0.0;
                    }
                }
                ++content_revision_;
                ++revision;
            }
        }
        if (density_.slot_revisions[update.frame_slot_index] != content_revision_ || color_.slot_revisions[update.frame_slot_index] != content_revision_ || target_wind_.slot_revisions[update.frame_slot_index] != content_revision_ || estimated_wind_.slot_revisions[update.frame_slot_index] != content_revision_) write_visualization(update.frame_slot_index);
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        scene.set_document(plugin::Document{
            .update = {.enabled = true, .initial_running = false, .step_delta_seconds = 1.0 / 60.0},
            .navigation_target = {
                .revision = 1u,
                .focus = {1.5F, 0.72F, 0.50F},
                .bounds_minimum = {-0.35F, -0.15F, -0.45F},
                .bounds_maximum = {3.35F, 1.65F, 1.25F},
                .navigation_up = {0.0F, 1.0F, 0.0F},
            },
            .active_camera_name = "Overview",
            .materials = {{
                .name = "Smoke Comparison",
                .base_color = {1.0F, 1.0F, 1.0F, 0.90F},
                .density = {.channel_name = "density", .component = 0u, .scale = 1.0F, .bias = 0.0F, .enabled = true},
                .color = {.channel_name = "color", .component = 0u, .scale = 1.0F, .bias = 0.0F, .enabled = true},
            }},
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo) const {
        const std::array<std::uint32_t, 3u>& resolution = task_.model.configuration.resolution;
        plugin::Document document{
            .cameras = {overview_camera()},
            .volumes = {{
                .name = "Target, Estimated, and Difference Smoke",
                .dimensions = {3u * resolution[0], resolution[1], resolution[2]},
                .origin = {0.0F, 0.0F, 0.0F},
                .voxel_size = {task_.model.configuration.cell_size, task_.model.configuration.cell_size, task_.model.configuration.cell_size},
                .channels = {
                    {.name = "density", .format = plugin::VolumeChannelFormat::float32, .buffer_id = density_.allocation.resource_id, .device_pointer = reinterpret_cast<std::uintptr_t>(density_.mapped_buffers[current_frame_slot_]), .ready_event = reinterpret_cast<std::uintptr_t>(density_.ready_events[current_frame_slot_]), .source_byte_size = density_value_bytes_},
                    {.name = "color", .format = plugin::VolumeChannelFormat::float32x3, .buffer_id = color_.allocation.resource_id, .device_pointer = reinterpret_cast<std::uintptr_t>(color_.mapped_buffers[current_frame_slot_]), .ready_event = reinterpret_cast<std::uintptr_t>(color_.ready_events[current_frame_slot_]), .source_byte_size = color_.allocation.byte_size},
                },
                .material_name = "Smoke Comparison",
            }},
        };
        if (show_wind_) {
            document.viewport_segment_sets.push_back({.name = "Target Wind", .owner_name = "Overview", .segment_count = 3u, .buffer_id = target_wind_.allocation.resource_id, .source_byte_size = target_wind_.allocation.byte_size, .width = wind_width_});
            document.viewport_segment_sets.push_back({.name = "Estimated Wind", .owner_name = "Overview", .segment_count = 3u, .buffer_id = estimated_wind_.allocation.resource_id, .source_byte_size = estimated_wind_.allocation.byte_size, .width = wind_width_});
        }
        scene.set_document(std::move(document));
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        const WindTrajectoryOptimizationMetrics& metrics = task_.metrics;
        const std::size_t control_step = std::min<std::size_t>(trajectory_step_, trajectory_steps - 1u);
        const Vector3 target_wind = interpolate_wind(task_.target_keyframes, control_step, trajectory_steps);
        const Vector3 estimated_wind = interpolate_wind(task_.estimated_keyframes, control_step, trajectory_steps);
        const std::string phase = update_running_ ? (trajectory_playback_ ? "Optimizing + Playing Trajectory" : "Optimizing") : (trajectory_playback_ ? "Playing Trajectory" : "Paused");
        controls
            .phase(phase)
            .headline("Differentiable local-wind trajectory optimization")
            .message("Left: target density. Center: estimated density. Right: absolute density difference. Bottom playback advances optimization; trajectory playback and frame selection inspect the cached forward simulation.")
            .metric("iteration", "Iteration", std::to_string(metrics.iteration), "optimization")
            .metric("loss", "Trajectory Loss", std::format("{:.9e}", metrics.loss), "optimization")
            .metric("loss_ratio", "Final / Initial Loss", std::format("{:.3e}", metrics.loss_ratio), "optimization")
            .metric("keyframe_error", "Keyframe Relative Error", std::format("{:.3e}", metrics.keyframe_relative_error), "optimization")
            .metric("gradient_norm", "Gradient L2 Norm", std::format("{:.9e}", metrics.gradient_norm), "optimization")
            .metric("target_wind", "Target Wind X / Z", std::format("{:+.5f} / {:+.5f}", target_wind.x, target_wind.z), "optimization")
            .metric("estimated_wind", "Estimated Wind X / Z", std::format("{:+.5f} / {:+.5f}", estimated_wind.x, estimated_wind.z), "optimization")
            .metric("duration", "Trajectory Duration", std::format("{:.3f} s", static_cast<double>(trajectory_steps) * task_.model.configuration.time_step), "optimization")
            .metric("grid", "Grid", std::format("{} x {} x {}", task_.model.configuration.resolution[0], task_.model.configuration.resolution[1], task_.model.configuration.resolution[2]), "physics")
            .metric("optimization_time", "Optimization Time", std::format("{:.3f} ms", optimization_milliseconds_), "optimization")
            .metric("trajectory_frame", "Trajectory Frame", std::format("{} / {}", trajectory_step_, trajectory_steps), "display")
            .metric("physical_time", "Physical Time", std::format("{:.3f} s", static_cast<double>(trajectory_step_) * task_.model.configuration.time_step), "display")
            .metric("target_density", "Target Density Mean / Max", std::format("{:.5f} / {:.5f}", target_density_mean_, target_density_maximum_), "display")
            .metric("estimated_density", "Estimated Density Mean / Max", std::format("{:.5f} / {:.5f}", estimated_density_mean_, estimated_density_maximum_), "display")
            .metric("slot", "Frame Slot", std::to_string(current_frame_slot_), "display")
            .unsigned_setting("trajectory_frame", trajectory_step_, 0u, trajectory_steps, 1u)
            .setting("play_trajectory", trajectory_playback_ ? "true" : "false")
            .setting("loop_trajectory", loop_trajectory_ ? "true" : "false")
            .setting("show_target", show_target_ ? "true" : "false")
            .setting("show_estimate", show_estimate_ ? "true" : "false")
            .setting("show_difference", show_difference_ ? "true" : "false")
            .setting("show_wind", show_wind_ ? "true" : "false")
            .setting("trajectory_fps", std::format("{}", trajectory_fps_))
            .setting("smoke_opacity_scale", std::format("{}", smoke_opacity_scale_))
            .setting("difference_opacity_scale", std::format("{}", difference_opacity_scale_))
            .setting("wind_width", std::format("{}", wind_width_))
            .setting("wind_scale", std::format("{}", wind_scale_))
            .enable("reset");
    }

    void Project::write_visualization(const std::uint32_t frame_slot_index) {
        const cudaStream_t stream = static_cast<cudaStream_t>(task_.context.resource.native_stream);
        const std::array<std::uint32_t, 3u>& resolution = task_.model.configuration.resolution;
        const State& target_state = task_.target_trajectory.states[trajectory_step_];
        const State& estimated_state = task_.estimated_trajectory.states[trajectory_step_];
        const std::size_t control_step = std::min<std::size_t>(trajectory_step_, trajectory_steps - 1u);
        const Vector3 target_wind = interpolate_wind(task_.target_keyframes, control_step, trajectory_steps);
        const Vector3 estimated_wind = interpolate_wind(task_.estimated_keyframes, control_step, trajectory_steps);
        double* const device_statistics = reinterpret_cast<double*>(static_cast<std::byte*>(density_.mapped_buffers[frame_slot_index]) + density_value_bytes_);
        task_.context.resource.zero(device_statistics, 4u * sizeof(double));
        visualization_cuda::launch_volume(stream, resolution[0], resolution[1], resolution[2], target_state.density.values.data, estimated_state.density.values.data, show_target_, show_estimate_, show_difference_, smoke_opacity_scale_, difference_opacity_scale_, static_cast<float*>(density_.mapped_buffers[frame_slot_index]), static_cast<float*>(color_.mapped_buffers[frame_slot_index]), device_statistics);
        visualization_cuda::launch_wind_arrow(stream, 0.5F, 0.75F, -0.20F, target_wind.x, target_wind.z, wind_scale_, wind_width_, visualization_cuda::WindStyle::target, target_wind_.mapped_buffers[frame_slot_index]);
        visualization_cuda::launch_wind_arrow(stream, 1.5F, 0.75F, -0.20F, estimated_wind.x, estimated_wind.z, wind_scale_, wind_width_, visualization_cuda::WindStyle::estimated, estimated_wind_.mapped_buffers[frame_slot_index]);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error(std::format("smoke visualization kernel launch failed: {}", cudaGetErrorString(status)));
        std::array<double, 4u> statistics{};
        task_.context.resource.copy_to_host(statistics.data(), device_statistics, 4u * sizeof(double));
        task_.context.resource.synchronize();
        const double cell_count = static_cast<double>(resolution[0]) * resolution[1] * resolution[2];
        target_density_mean_ = statistics[0] / cell_count;
        target_density_maximum_ = statistics[1];
        estimated_density_mean_ = statistics[2] / cell_count;
        estimated_density_maximum_ = statistics[3];
        density_.record_ready(frame_slot_index, stream);
        color_.record_ready(frame_slot_index, stream);
        target_wind_.record_ready(frame_slot_index, stream);
        estimated_wind_.record_ready(frame_slot_index, stream);
        density_.slot_revisions[frame_slot_index] = content_revision_;
        color_.slot_revisions[frame_slot_index] = content_revision_;
        target_wind_.slot_revisions[frame_slot_index] = content_revision_;
        estimated_wind_.slot_revisions[frame_slot_index] = content_revision_;
    }

} // namespace xayah::smoke::examples::wind_trajectory_optimization::project

extern "C" XAYAH_SMOKE_WIND_TRAJECTORY_OPTIMIZATION_PLUGIN_EXPORT auto spectra_scene_plugin_v21() -> decltype(xayah::spectra::plugin::export_plugin<xayah::smoke::examples::wind_trajectory_optimization::project::Project>()) {
    return xayah::spectra::plugin::export_plugin<xayah::smoke::examples::wind_trajectory_optimization::project::Project>();
}
