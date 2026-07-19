#if defined(_WIN32)
#define XAYAH_WIND_TRAJECTORY_OPTIMIZATION_PLUGIN_EXPORT __declspec(dllexport)
#else
#define XAYAH_WIND_TRAJECTORY_OPTIMIZATION_PLUGIN_EXPORT __attribute__((visibility("default")))
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
import xayah.cloth.data;
import xayah.examples.cloth.wind_trajectory_optimization;
import xayah.cloth.model;
import xayah.spectra.plugin;

namespace xayah::cloth::examples::wind_trajectory_optimization::project {

    namespace plugin = spectra::plugin;

    namespace {

        constexpr std::uint64_t segment_bytes = 48u;

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
            std::uint32_t rows{16u};
            std::uint32_t columns{16u};
            float width{2.0F};
            float height{2.0F};
            float mass{0.05F};
            float stretch_stiffness{400.0F};
            float stretch_damping{1.0F};
            float bending_stiffness{5.0F};
            float bending_damping{0.1F};
            float gravity_y{-9.81F};
            float time_step{1.0F / 240.0F};
            std::uint32_t trajectory_steps{120u};
            float adam_learning_rate{0.02F};
            std::uint32_t iterations_per_update{1u};
        };

        [[nodiscard]] Configuration make_configuration(const ProjectOptions& options);
        [[nodiscard]] plugin::Camera overview_camera(float width, float height);
        [[nodiscard]] Vector3 interpolate_wind(std::span<const Vector3> keyframes, std::size_t control_step, std::size_t trajectory_steps);

    } // namespace

    struct Project final {
        std::uint64_t revision{1u};

        Project(ProjectOptions options, Configuration configuration, std::shared_ptr<plugin::HostServices> host_services);
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
        Parameters visualization_parameters_;
        ExternalBuffer target_stretch_segments_{};
        ExternalBuffer estimated_stretch_segments_{};
        ExternalBuffer estimated_bending_segments_{};
        ExternalBuffer target_wind_segments_{};
        ExternalBuffer estimated_wind_segments_{};
        std::uint64_t content_revision_{1u};
        float optimization_milliseconds_{};
        std::uint32_t current_frame_slot_{};
        std::uint32_t trajectory_step_;
        bool update_running_{};
        bool reset_pending_{};
        bool show_target_{true};
        bool show_estimate_{true};
        bool show_wind_{true};
        bool show_bending_{};
        float stretch_width_{2.0F};
        float bending_width_{1.0F};
        float wind_width_{3.0F};
        float wind_scale_{2.0F};
        float strain_range_{0.10F};
    };

    namespace {

        Configuration make_configuration(const ProjectOptions& options) {
            Configuration configuration{
                .rest_positions = std::vector<Vector3>(static_cast<std::size_t>(options.rows) * options.columns),
                .triangles = {},
                .anchors = std::vector<std::optional<Vector3>>(static_cast<std::size_t>(options.rows) * options.columns),
                .gravity = {.x = 0.0F, .y = options.gravity_y, .z = 0.0F},
                .time_step = options.time_step,
            };
            const float spacing_x = options.width / static_cast<float>(options.columns - 1u);
            const float spacing_z = options.height / static_cast<float>(options.rows - 1u);
            for (std::uint32_t row = 0u; row < options.rows; ++row) {
                for (std::uint32_t column = 0u; column < options.columns; ++column) {
                    const std::uint32_t vertex = row * options.columns + column;
                    configuration.rest_positions[vertex] = {.x = static_cast<float>(column) * spacing_x, .y = 0.0F, .z = static_cast<float>(row) * spacing_z};
                    if (row == 0u) configuration.anchors[vertex] = configuration.rest_positions[vertex];
                }
            }
            configuration.triangles.reserve(static_cast<std::size_t>(options.rows - 1u) * (options.columns - 1u) * 2u);
            for (std::uint32_t row = 0u; row + 1u < options.rows; ++row) {
                for (std::uint32_t column = 0u; column + 1u < options.columns; ++column) {
                    const std::uint32_t top_left = row * options.columns + column;
                    const std::uint32_t top_right = top_left + 1u;
                    const std::uint32_t bottom_left = top_left + options.columns;
                    const std::uint32_t bottom_right = bottom_left + 1u;
                    if ((row + column) % 2u == 0u) {
                        configuration.triangles.push_back({.first = top_left, .second = top_right, .third = bottom_right});
                        configuration.triangles.push_back({.first = top_left, .second = bottom_right, .third = bottom_left});
                    } else {
                        configuration.triangles.push_back({.first = top_left, .second = top_right, .third = bottom_left});
                        configuration.triangles.push_back({.first = top_right, .second = bottom_right, .third = bottom_left});
                    }
                }
            }
            return configuration;
        }

        plugin::Camera overview_camera(const float width, const float height) {
            const std::array<float, 3u> position{width * 0.5F, height * 0.45F, height * 2.4F};
            const std::array<float, 3u> target{width * 0.5F, -height * 0.45F, height * 0.5F};
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
            return plugin::Camera{
                .name = "Overview",
                .position = position,
                .right = right,
                .down = down,
                .forward = forward,
                .vertical_fov_degrees = 42.0F,
                .near_plane = 0.01F,
                .far_plane = height * 12.0F,
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

    Project::Project(ProjectOptions options, Configuration configuration, std::shared_ptr<plugin::HostServices> host_services)
        : options_(options), task_(std::move(configuration), WindTrajectoryOptimizationOptions{
              .mass = options_.mass,
              .stretch_stiffness = options_.stretch_stiffness,
              .stretch_damping = options_.stretch_damping,
              .bending_stiffness = options_.bending_stiffness,
              .bending_damping = options_.bending_damping,
              .trajectory_steps = options_.trajectory_steps,
              .adam_learning_rate = options_.adam_learning_rate,
          }), visualization_parameters_(task_.model.make_parameters(task_.context)), trajectory_step_(options_.trajectory_steps) {
        std::vector<float> stretch_rest_lengths(task_.model.topology.stretch_springs.size());
        for (std::size_t spring = 0u; spring < stretch_rest_lengths.size(); ++spring) stretch_rest_lengths[spring] = task_.model.topology.stretch_springs[spring].rest_length;
        task_.context.upload(stretch_rest_lengths, visualization_parameters_.stretch_rest_lengths);
        std::vector<float> bending_rest_lengths(task_.model.topology.bending_springs.size());
        for (std::size_t spring = 0u; spring < bending_rest_lengths.size(); ++spring) bending_rest_lengths[spring] = task_.model.topology.bending_springs[spring].rest_length;
        task_.context.upload(bending_rest_lengths, visualization_parameters_.bending_rest_lengths);
        target_stretch_segments_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, task_.model.topology.stretch_springs.size() * segment_bytes);
        estimated_stretch_segments_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, task_.model.topology.stretch_springs.size() * segment_bytes);
        estimated_bending_segments_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, task_.model.topology.bending_springs.size() * segment_bytes);
        target_wind_segments_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, 3u * segment_bytes);
        estimated_wind_segments_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, 3u * segment_bytes);
        for (std::size_t frame_slot = 0u; frame_slot < target_stretch_segments_.mapped_buffers.size(); ++frame_slot) write_visualization(static_cast<std::uint32_t>(frame_slot));
        task_.context.synchronize();
    }

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition = [] {
            plugin::PluginDefinition<Project> value{
                .id = "xayah.examples.cloth.wind_trajectory_optimization",
                .title = "CUDA Cloth Wind Trajectory Optimization",
                .open_action_label = "Open CUDA Cloth Wind Trajectory Optimization",
                .sections = {{.id = "optimization", .label = "Optimization"}, {.id = "physics", .label = "Physics"}, {.id = "display", .label = "Display"}},
                .open_options = {
                    plugin::unsigned_integer("rows", "Rows", 16u, "physics"),
                    plugin::unsigned_integer("columns", "Columns", 16u, "physics"),
                    plugin::float_value("width", "Width", 2.0F, "physics"),
                    plugin::float_value("height", "Height", 2.0F, "physics"),
                    plugin::float_value("mass", "Mass", 0.05F, "physics"),
                    plugin::float_value("stretch_stiffness", "Stretch Stiffness", 400.0F, "physics"),
                    plugin::float_value("stretch_damping", "Stretch Damping", 1.0F, "physics"),
                    plugin::float_value("bending_stiffness", "Bending Stiffness", 5.0F, "physics"),
                    plugin::float_value("bending_damping", "Bending Damping", 0.1F, "physics"),
                    plugin::float_value("gravity_y", "Gravity Y", -9.81F, "physics"),
                    plugin::float_value("time_step", "Time Step", 1.0F / 240.0F, "physics"),
                    plugin::unsigned_integer("trajectory_steps", "Trajectory Steps", 120u, "optimization"),
                    plugin::float_value("adam_learning_rate", "Adam Learning Rate", 0.02F, "optimization"),
                    plugin::unsigned_integer("iterations_per_update", "Iterations Per Update", 1u, "optimization"),
                },
                .actions = {plugin::action<Project>("reset", "Reset Optimization", "Restore zero wind keyframes and the Adam state on the next safe frame-slot update.", "optimization", [](Project& project) { project.reset_pending_ = true; })},
                .settings = {
                    plugin::unsigned_integer_setting<Project>("trajectory_frame", "Trajectory Frame", 120u, "display", 0u, 120u, 1u, [](Project& project, const std::uint64_t value) {
                        project.trajectory_step_ = static_cast<std::uint32_t>(value);
                        ++project.content_revision_;
                    }),
                    plugin::toggle<Project>("show_target", "Show Target", true, "display", [](Project& project, const bool value) {
                        project.show_target_ = value;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_estimate", "Show Estimate", true, "display", [](Project& project, const bool value) {
                        project.show_estimate_ = value;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_wind", "Show Wind", true, "display", [](Project& project, const bool value) {
                        project.show_wind_ = value;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_bending", "Show Bending", false, "display", [](Project& project, const bool value) {
                        project.show_bending_ = value;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("stretch_width", "Stretch Width", 2.0F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.stretch_width_ = value;
                        ++project.content_revision_;
                    }),
                    plugin::float_setting<Project>("bending_width", "Bending Width", 1.0F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.bending_width_ = value;
                        ++project.content_revision_;
                    }),
                    plugin::float_setting<Project>("wind_width", "Wind Width", 3.0F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.wind_width_ = value;
                        ++project.content_revision_;
                    }),
                    plugin::float_setting<Project>("wind_scale", "Wind Scale", 2.0F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.wind_scale_ = value;
                        ++project.content_revision_;
                    }),
                    plugin::float_setting<Project>("strain_range", "Strain Range", 0.10F, "display", 0.01F, 0.50F, 0.01F, [](Project& project, const float value) {
                        project.strain_range_ = value;
                        ++project.content_revision_;
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
            if (option.key == "rows") options.rows = static_cast<std::uint32_t>(std::stoul(option.value));
            else if (option.key == "columns") options.columns = static_cast<std::uint32_t>(std::stoul(option.value));
            else if (option.key == "width") options.width = std::stof(option.value);
            else if (option.key == "height") options.height = std::stof(option.value);
            else if (option.key == "mass") options.mass = std::stof(option.value);
            else if (option.key == "stretch_stiffness") options.stretch_stiffness = std::stof(option.value);
            else if (option.key == "stretch_damping") options.stretch_damping = std::stof(option.value);
            else if (option.key == "bending_stiffness") options.bending_stiffness = std::stof(option.value);
            else if (option.key == "bending_damping") options.bending_damping = std::stof(option.value);
            else if (option.key == "gravity_y") options.gravity_y = std::stof(option.value);
            else if (option.key == "time_step") options.time_step = std::stof(option.value);
            else if (option.key == "trajectory_steps") options.trajectory_steps = static_cast<std::uint32_t>(std::stoul(option.value));
            else if (option.key == "adam_learning_rate") options.adam_learning_rate = std::stof(option.value);
            else if (option.key == "iterations_per_update") options.iterations_per_update = static_cast<std::uint32_t>(std::stoul(option.value));
        }
        Configuration configuration = make_configuration(options);
        return Project(options, std::move(configuration), std::move(context.host_services));
    }

    void Project::update(const plugin::UpdateInfo& update) {
        this->current_frame_slot_ = update.frame_slot_index;
        this->update_running_ = update.update_running;
        if (this->reset_pending_) {
            this->task_.reset();
            this->reset_pending_ = false;
            ++this->content_revision_;
        }
        if (update.update_delta_seconds > 0.0) {
            const auto begin = std::chrono::steady_clock::now();
            for (std::uint32_t iteration = 0u; iteration < this->options_.iterations_per_update; ++iteration) this->task_.optimize_step();
            this->optimization_milliseconds_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
            ++this->content_revision_;
        }
        if (this->target_stretch_segments_.slot_revisions[update.frame_slot_index] != this->content_revision_ || this->estimated_stretch_segments_.slot_revisions[update.frame_slot_index] != this->content_revision_ || this->estimated_bending_segments_.slot_revisions[update.frame_slot_index] != this->content_revision_ || this->target_wind_segments_.slot_revisions[update.frame_slot_index] != this->content_revision_ || this->estimated_wind_segments_.slot_revisions[update.frame_slot_index] != this->content_revision_) this->write_visualization(update.frame_slot_index);
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        scene.set_document(plugin::Document{
            .update = {.enabled = true, .initial_running = false, .step_delta_seconds = 1.0 / 60.0},
            .navigation_target = {
                .revision = 1u,
                .focus = {this->options_.width * 0.5F, -this->options_.height * 0.45F, this->options_.height * 0.5F},
                .bounds_minimum = {-this->options_.width * 0.75F, -this->options_.height * 1.25F, -this->options_.height * 0.1F},
                .bounds_maximum = {this->options_.width * 1.1F, this->options_.height * 0.25F, this->options_.height * 1.1F},
                .navigation_up = {0.0F, 1.0F, 0.0F},
            },
            .active_camera_name = "Overview",
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo) const {
        plugin::Document document{.cameras = {overview_camera(this->options_.width, this->options_.height)}};
        if (this->show_target_) document.viewport_segment_sets.push_back({.name = "Target Stretch Springs", .owner_name = "Overview", .segment_count = this->task_.model.topology.stretch_springs.size(), .buffer_id = this->target_stretch_segments_.allocation.resource_id, .source_byte_size = this->target_stretch_segments_.allocation.byte_size, .width = 1.0F});
        if (this->show_estimate_) document.viewport_segment_sets.push_back({.name = "Estimated Stretch Springs", .owner_name = "Overview", .segment_count = this->task_.model.topology.stretch_springs.size(), .buffer_id = this->estimated_stretch_segments_.allocation.resource_id, .source_byte_size = this->estimated_stretch_segments_.allocation.byte_size, .width = this->stretch_width_});
        if (this->show_bending_) document.viewport_segment_sets.push_back({.name = "Estimated Bending Springs", .owner_name = "Overview", .segment_count = this->task_.model.topology.bending_springs.size(), .buffer_id = this->estimated_bending_segments_.allocation.resource_id, .source_byte_size = this->estimated_bending_segments_.allocation.byte_size, .width = this->bending_width_});
        if (this->show_wind_) {
            document.viewport_segment_sets.push_back({.name = "Target Wind", .owner_name = "Overview", .segment_count = 3u, .buffer_id = this->target_wind_segments_.allocation.resource_id, .source_byte_size = this->target_wind_segments_.allocation.byte_size, .width = this->wind_width_});
            document.viewport_segment_sets.push_back({.name = "Estimated Wind", .owner_name = "Overview", .segment_count = 3u, .buffer_id = this->estimated_wind_segments_.allocation.resource_id, .source_byte_size = this->estimated_wind_segments_.allocation.byte_size, .width = this->wind_width_});
        }
        scene.set_document(std::move(document));
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        const WindTrajectoryOptimizationMetrics& metrics = this->task_.metrics;
        const std::size_t control_step = std::min<std::size_t>(this->trajectory_step_, this->options_.trajectory_steps - 1u);
        const Vector3 target_wind = interpolate_wind(this->task_.target_keyframes, control_step, this->options_.trajectory_steps);
        const Vector3 estimated_wind = interpolate_wind(this->task_.estimated_keyframes, control_step, this->options_.trajectory_steps);
        controls
            .phase(this->update_running_ ? "Optimizing" : "Paused")
            .headline("Differentiable wind-trajectory optimization")
            .message("The full target and estimated trajectories, loss seeds, VJP, and segment instances remain on the GPU.")
            .metric("iteration", "Iteration", std::to_string(metrics.iteration), "optimization")
            .metric("loss", "Trajectory Loss", std::format("{:.9e}", metrics.loss), "optimization")
            .metric("loss_ratio", "Final / Initial Loss", std::format("{:.3e}", metrics.loss_ratio), "optimization")
            .metric("keyframe_error", "Keyframe Relative Error", std::format("{:.3e}", metrics.keyframe_relative_error), "optimization")
            .metric("gradient_norm", "Gradient L2 Norm", std::format("{:.9e}", metrics.gradient_norm), "optimization")
            .metric("target_wind", "Target Wind X / Z", std::format("{:+.5f} / {:+.5f}", target_wind.x, target_wind.z), "optimization")
            .metric("estimated_wind", "Estimated Wind X / Z", std::format("{:+.5f} / {:+.5f}", estimated_wind.x, estimated_wind.z), "optimization")
            .metric("duration", "Trajectory Duration", std::format("{:.4f} s", static_cast<double>(this->options_.trajectory_steps) * this->options_.time_step), "optimization")
            .metric("grid", "Grid", std::format("{} x {}", this->options_.rows, this->options_.columns), "physics")
            .metric("optimization_time", "Optimization Time", std::format("{:.3f} ms", this->optimization_milliseconds_), "optimization")
            .metric("trajectory_frame", "Trajectory Frame", std::format("{} / {}", this->trajectory_step_, this->options_.trajectory_steps), "display")
            .metric("physical_time", "Physical Time", std::format("{:.4f} s", static_cast<double>(this->trajectory_step_) * this->options_.time_step), "display")
            .metric("slot", "Frame Slot", std::to_string(this->current_frame_slot_), "display")
            .unsigned_setting("trajectory_frame", this->trajectory_step_, 0u, this->options_.trajectory_steps, 1u)
            .setting("show_target", this->show_target_ ? "true" : "false")
            .setting("show_estimate", this->show_estimate_ ? "true" : "false")
            .setting("show_wind", this->show_wind_ ? "true" : "false")
            .setting("show_bending", this->show_bending_ ? "true" : "false")
            .setting("stretch_width", std::format("{}", this->stretch_width_))
            .setting("bending_width", std::format("{}", this->bending_width_))
            .setting("wind_width", std::format("{}", this->wind_width_))
            .setting("wind_scale", std::format("{}", this->wind_scale_))
            .setting("strain_range", std::format("{}", this->strain_range_))
            .enable("reset");
    }

    void Project::write_visualization(const std::uint32_t frame_slot_index) {
        ExecutionContext& context = this->task_.context;
        const Model& model = this->task_.model;
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource.native_stream);
        const DeviceTopology& topology = context.device_topology;
        const ::xayah::cloth::State& target_state = this->task_.target_trajectory.states[this->trajectory_step_];
        const ::xayah::cloth::State& estimated_state = this->task_.estimated_trajectory.states[this->trajectory_step_];
        const std::size_t control_step = std::min<std::size_t>(this->trajectory_step_, this->options_.trajectory_steps - 1u);
        const Vector3 target_wind = interpolate_wind(this->task_.target_keyframes, control_step, this->options_.trajectory_steps);
        const Vector3 estimated_wind = interpolate_wind(this->task_.estimated_keyframes, control_step, this->options_.trajectory_steps);
        visualization_cuda::launch_segments(
            stream,
            static_cast<std::uint32_t>(model.topology.stretch_springs.size()),
            target_state.positions.x.data,
            target_state.positions.y.data,
            target_state.positions.z.data,
            topology.stretch.first.data,
            topology.stretch.second.data,
            this->visualization_parameters_.stretch_rest_lengths.data,
            1.0F,
            this->strain_range_,
            visualization_cuda::SegmentStyle::target,
            this->target_stretch_segments_.mapped_buffers[frame_slot_index]);
        visualization_cuda::launch_segments(
            stream,
            static_cast<std::uint32_t>(model.topology.stretch_springs.size()),
            estimated_state.positions.x.data,
            estimated_state.positions.y.data,
            estimated_state.positions.z.data,
            topology.stretch.first.data,
            topology.stretch.second.data,
            this->visualization_parameters_.stretch_rest_lengths.data,
            this->stretch_width_,
            this->strain_range_,
            visualization_cuda::SegmentStyle::estimate,
            this->estimated_stretch_segments_.mapped_buffers[frame_slot_index]);
        visualization_cuda::launch_segments(
            stream,
            static_cast<std::uint32_t>(model.topology.bending_springs.size()),
            estimated_state.positions.x.data,
            estimated_state.positions.y.data,
            estimated_state.positions.z.data,
            topology.bending.first.data,
            topology.bending.second.data,
            this->visualization_parameters_.bending_rest_lengths.data,
            this->bending_width_,
            this->strain_range_,
            visualization_cuda::SegmentStyle::bending,
            this->estimated_bending_segments_.mapped_buffers[frame_slot_index]);
        visualization_cuda::launch_wind_arrow(
            stream,
            -0.42F * this->options_.width,
            -0.12F * this->options_.height,
            0.50F * this->options_.height,
            target_wind.x,
            target_wind.z,
            this->wind_scale_,
            this->wind_width_,
            visualization_cuda::SegmentStyle::target_wind,
            this->target_wind_segments_.mapped_buffers[frame_slot_index]);
        visualization_cuda::launch_wind_arrow(
            stream,
            -0.42F * this->options_.width,
            -0.12F * this->options_.height,
            0.50F * this->options_.height,
            estimated_wind.x,
            estimated_wind.z,
            this->wind_scale_,
            this->wind_width_,
            visualization_cuda::SegmentStyle::estimated_wind,
            this->estimated_wind_segments_.mapped_buffers[frame_slot_index]);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error(std::format("cloth visualization kernel launch failed: {}", cudaGetErrorString(status)));
        this->target_stretch_segments_.slot_revisions[frame_slot_index] = this->content_revision_;
        this->estimated_stretch_segments_.slot_revisions[frame_slot_index] = this->content_revision_;
        this->estimated_bending_segments_.slot_revisions[frame_slot_index] = this->content_revision_;
        this->target_wind_segments_.slot_revisions[frame_slot_index] = this->content_revision_;
        this->estimated_wind_segments_.slot_revisions[frame_slot_index] = this->content_revision_;
    }

} // namespace xayah::cloth::examples::wind_trajectory_optimization::project

extern "C" XAYAH_WIND_TRAJECTORY_OPTIMIZATION_PLUGIN_EXPORT auto spectra_scene_plugin_v21() -> decltype(xayah::spectra::plugin::export_plugin<xayah::cloth::examples::wind_trajectory_optimization::project::Project>()) {
    return xayah::spectra::plugin::export_plugin<xayah::cloth::examples::wind_trajectory_optimization::project::Project>();
}
