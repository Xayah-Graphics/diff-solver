#if defined(_WIN32)
#define XAYAH_CLOTH_FORWARD_PLUGIN_EXPORT __declspec(dllexport)
#else
#define XAYAH_CLOTH_FORWARD_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime_api.h>

#include "../visualization.h"

import std;
import xayah.cloth.data;
import xayah.examples.cloth.forward;
import xayah.spectra.plugin;

namespace xayah::cloth::examples::forward::project {

    namespace plugin = spectra::plugin;

    namespace {

        constexpr std::uint64_t segment_bytes = 48u;

        struct ExternalBuffer {
            ExternalBuffer() = default;

            ~ExternalBuffer() noexcept {
                for (void* const mapped_buffer : mapped_buffers) if (mapped_buffer != nullptr && cudaFree(mapped_buffer) != cudaSuccess) std::terminate();
                for (const cudaExternalMemory_t external_memory : external_memories_) if (external_memory != nullptr && cudaDestroyExternalMemory(external_memory) != cudaSuccess) std::terminate();
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
                external_memories.reserve(next_allocation.slots.size());
                next_mapped_buffers.reserve(next_allocation.slots.size());
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
                        external_memories.push_back(external_memory);
                        next_mapped_buffers.push_back(mapped_buffer);
                    }
                } catch (...) {
                    for (plugin::GpuBufferSlotAllocation& slot : next_allocation.slots) if (slot.handle != 0u) close_imported_handle(slot);
                    for (void* const mapped_buffer : next_mapped_buffers) if (mapped_buffer != nullptr) static_cast<void>(cudaFree(mapped_buffer));
                    for (const cudaExternalMemory_t external_memory : external_memories) if (external_memory != nullptr) static_cast<void>(cudaDestroyExternalMemory(external_memory));
                    host_services->release_gpu_buffer(next_allocation.resource_id);
                    throw;
                }
                host_services_ = std::move(host_services);
                allocation = std::move(next_allocation);
                external_memories_ = std::move(external_memories);
                mapped_buffers = std::move(next_mapped_buffers);
                slot_revisions.assign(mapped_buffers.size(), 0u);
            }

            plugin::GpuBufferAllocation allocation{};
            std::vector<void*> mapped_buffers{};
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

        plugin::Camera overview_camera() {
            const std::array<float, 3u> position{4.35F, -0.15F, 5.80F};
            const std::array<float, 3u> target{1.50F, -1.00F, 0.00F};
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
                .vertical_fov_degrees = 34.0F,
                .near_plane = 0.01F,
                .far_plane = 30.0F,
            };
        }

    } // namespace

    struct Project final {
        std::uint64_t revision{1u};

        explicit Project(std::shared_ptr<plugin::HostServices> host_services);
        ~Project() noexcept;

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

        ForwardSimulation simulation_{};
        float* stretch_rest_lengths_{};
        float* bending_rest_lengths_{};
        ExternalBuffer stretch_segments_{};
        ExternalBuffer bending_segments_{};
        ExternalBuffer load_segments_{};
        std::uint64_t content_revision_{1u};
        std::uint32_t current_frame_slot_{};
        bool update_running_{};
        bool reset_pending_{};
        bool show_stretch_{true};
        bool show_bending_{};
        bool show_load_{true};
        float stretch_width_{1.5F};
        float bending_width_{1.0F};
        float load_width_{3.0F};
        float load_scale_{0.08F};
        float strain_range_{0.10F};
    };

    Project::Project(std::shared_ptr<plugin::HostServices> host_services)
        : stretch_rest_lengths_(static_cast<float*>(simulation_.context.resource.allocate(simulation_.model.topology.stretch_springs.size() * sizeof(float)))), bending_rest_lengths_(static_cast<float*>(simulation_.context.resource.allocate(simulation_.model.topology.bending_springs.size() * sizeof(float)))) {
        std::vector<float> stretch_rest_lengths(simulation_.model.topology.stretch_springs.size());
        for (std::size_t spring = 0u; spring < stretch_rest_lengths.size(); ++spring) stretch_rest_lengths[spring] = simulation_.model.topology.stretch_springs[spring].rest_length;
        simulation_.context.resource.copy_from_host(stretch_rest_lengths_, stretch_rest_lengths.data(), stretch_rest_lengths.size() * sizeof(float));
        std::vector<float> bending_rest_lengths(simulation_.model.topology.bending_springs.size());
        for (std::size_t spring = 0u; spring < bending_rest_lengths.size(); ++spring) bending_rest_lengths[spring] = simulation_.model.topology.bending_springs[spring].rest_length;
        simulation_.context.resource.copy_from_host(bending_rest_lengths_, bending_rest_lengths.data(), bending_rest_lengths.size() * sizeof(float));
        stretch_segments_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, simulation_.model.topology.stretch_springs.size() * segment_bytes);
        bending_segments_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, simulation_.model.topology.bending_springs.size() * segment_bytes);
        load_segments_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, 9u * segment_bytes);
        for (std::size_t frame_slot = 0u; frame_slot < stretch_segments_.mapped_buffers.size(); ++frame_slot) write_visualization(static_cast<std::uint32_t>(frame_slot));
        simulation_.context.synchronize();
    }

    Project::~Project() noexcept {
        simulation_.context.resource.release(stretch_rest_lengths_);
        simulation_.context.resource.release(bending_rest_lengths_);
    }

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition = [] {
            plugin::PluginDefinition<Project> value{
                .id = "xayah.examples.cloth.forward",
                .title = "CUDA Prescribed Traveling Load Cloth",
                .open_action_label = "Open CUDA Prescribed Traveling Load Cloth",
                .sections = {{.id = "simulation", .label = "Simulation"}, {.id = "display", .label = "Display"}},
                .actions = {plugin::action<Project>("reset", "Reset Simulation", "Restore the stationary rest state and physical step zero on the next safe frame-slot update.", "simulation", [](Project& project) { project.reset_pending_ = true; })},
                .settings = {
                    plugin::toggle<Project>("show_stretch", "Show Stretch", true, "display", [](Project& project, const bool value) {
                        project.show_stretch_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_bending", "Show Bending", false, "display", [](Project& project, const bool value) {
                        project.show_bending_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_load", "Show Load", true, "display", [](Project& project, const bool value) {
                        project.show_load_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("stretch_width", "Stretch Width", 1.5F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.stretch_width_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("bending_width", "Bending Width", 1.0F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.bending_width_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("load_width", "Load Width", 3.0F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.load_width_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("load_scale", "Load Scale", 0.08F, "display", 0.005F, 0.15F, 0.005F, [](Project& project, const float value) {
                        project.load_scale_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("strain_range", "Strain Range", 0.10F, "display", 0.01F, 0.50F, 0.01F, [](Project& project, const float value) {
                        project.strain_range_ = value;
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
        return Project(std::move(context.host_services));
    }

    void Project::update(const plugin::UpdateInfo& update) {
        current_frame_slot_ = update.frame_slot_index;
        update_running_ = update.update_running;
        if (reset_pending_) {
            simulation_.reset();
            reset_pending_ = false;
            ++content_revision_;
            ++revision;
        }
        if (update.update_delta_seconds > 0.0) {
            simulation_.step();
            ++content_revision_;
            ++revision;
        }
        if (stretch_segments_.slot_revisions[update.frame_slot_index] != content_revision_ || bending_segments_.slot_revisions[update.frame_slot_index] != content_revision_ || load_segments_.slot_revisions[update.frame_slot_index] != content_revision_) write_visualization(update.frame_slot_index);
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        scene.set_document(plugin::Document{
            .update = {.enabled = true, .initial_running = false, .step_delta_seconds = simulation_.options.time_step},
            .navigation_target = {
                .revision = 1u,
                .focus = {1.50F, -1.00F, 0.00F},
                .bounds_minimum = {-0.25F, -2.40F, -1.80F},
                .bounds_maximum = {3.30F, 0.45F, 1.80F},
                .navigation_up = {0.0F, 1.0F, 0.0F},
            },
            .active_camera_name = "Overview",
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo) const {
        plugin::Document document{.cameras = {overview_camera()}};
        if (show_stretch_) document.viewport_segment_sets.push_back({.name = "Stretch Springs", .owner_name = "Overview", .segment_count = simulation_.model.topology.stretch_springs.size(), .buffer_id = stretch_segments_.allocation.resource_id, .source_byte_size = stretch_segments_.allocation.byte_size, .width = stretch_width_});
        if (show_bending_) document.viewport_segment_sets.push_back({.name = "Bending Springs", .owner_name = "Overview", .segment_count = simulation_.model.topology.bending_springs.size(), .buffer_id = bending_segments_.allocation.resource_id, .source_byte_size = bending_segments_.allocation.byte_size, .width = bending_width_});
        if (show_load_) document.viewport_segment_sets.push_back({.name = "Prescribed Traveling Load", .owner_name = "Overview", .segment_count = 9u, .buffer_id = load_segments_.allocation.resource_id, .source_byte_size = load_segments_.allocation.byte_size, .width = load_width_});
        scene.set_document(std::move(document));
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        const ForwardSimulationMetrics& metrics = simulation_.metrics;
        controls
            .phase(update_running_ ? "Forward Simulation" : "Paused")
            .headline("64 x 96 prescribed-traveling-load forward cloth")
            .message("Bottom playback advances physical time directly. This gravity-free scene isolates a prescribed mass-independent traveling normal load; it is not an aerodynamic model.")
            .metric("grid", "Grid", std::format("{} x {}", simulation_.options.rows, simulation_.options.columns), "simulation")
            .metric("step", "Physical Step", std::to_string(metrics.step), "simulation")
            .metric("time", "Physical Time", std::format("{:.3f} s", metrics.physical_time), "simulation")
            .metric("stretch_material", "Stretch Stiffness / Damping", std::format("{:.1f} / {:.2f}", simulation_.options.stretch_stiffness, simulation_.options.stretch_damping), "simulation")
            .metric("bending_material", "Bending Stiffness / Damping", std::format("{:.1f} / {:.2f}", simulation_.options.bending_stiffness, simulation_.options.bending_damping), "simulation")
            .metric("load_quarter", "Load u=0.25", std::format("{:+.5f} m/s^2", metrics.sampled_load_accelerations[0]), "simulation")
            .metric("load_half", "Load u=0.50", std::format("{:+.5f} m/s^2", metrics.sampled_load_accelerations[1]), "simulation")
            .metric("load_three_quarters", "Load u=0.75", std::format("{:+.5f} m/s^2", metrics.sampled_load_accelerations[2]), "simulation")
            .metric("free_edge_position", "Free Edge Mean Position", std::format("[{:+.5f}, {:+.5f}, {:+.5f}] m", metrics.free_edge_mean_position.x, metrics.free_edge_mean_position.y, metrics.free_edge_mean_position.z), "simulation")
            .metric("free_edge_displacement", "Free Edge Mean Displacement", std::format("[{:+.5f}, {:+.5f}, {:+.5f}] m", metrics.free_edge_mean_displacement.x, metrics.free_edge_mean_displacement.y, metrics.free_edge_mean_displacement.z), "simulation")
            .metric("velocity", "Maximum Velocity", std::format("{:.6f} m/s", metrics.maximum_velocity), "simulation")
            .metric("kinetic", "Kinetic Energy", std::format("{:.7f} J", metrics.kinetic_energy), "simulation")
            .metric("stretch_strain", "Maximum Stretch Strain", std::format("{:.6f}", metrics.maximum_absolute_stretch_strain), "simulation")
            .metric("bending_strain", "Maximum Bending Strain", std::format("{:.6f}", metrics.maximum_absolute_bending_strain), "simulation")
            .metric("step_time", "Forward Step", std::format("{:.3f} ms", metrics.step_milliseconds), "simulation")
            .metric("average_step_time", "Average Step", std::format("{:.3f} ms", metrics.average_step_milliseconds), "simulation")
            .metric("slot", "Frame Slot", std::to_string(current_frame_slot_), "display")
            .setting("show_stretch", show_stretch_ ? "true" : "false")
            .setting("show_bending", show_bending_ ? "true" : "false")
            .setting("show_load", show_load_ ? "true" : "false")
            .setting("stretch_width", std::format("{}", stretch_width_))
            .setting("bending_width", std::format("{}", bending_width_))
            .setting("load_width", std::format("{}", load_width_))
            .setting("load_scale", std::format("{}", load_scale_))
            .setting("strain_range", std::format("{}", strain_range_))
            .enable("reset");
    }

    void Project::write_visualization(const std::uint32_t frame_slot_index) {
        const cudaStream_t stream = static_cast<cudaStream_t>(simulation_.context.resource.native_stream);
        const DeviceTopology& topology = simulation_.context.device_topology;
        visualization_cuda::launch_segments(
            stream,
            static_cast<std::uint32_t>(simulation_.model.topology.stretch_springs.size()),
            simulation_.current_state.positions.x.data,
            simulation_.current_state.positions.y.data,
            simulation_.current_state.positions.z.data,
            topology.stretch.first.data,
            topology.stretch.second.data,
            stretch_rest_lengths_,
            stretch_width_,
            strain_range_,
            visualization_cuda::SegmentStyle::estimate,
            stretch_segments_.mapped_buffers[frame_slot_index]);
        visualization_cuda::launch_segments(
            stream,
            static_cast<std::uint32_t>(simulation_.model.topology.bending_springs.size()),
            simulation_.current_state.positions.x.data,
            simulation_.current_state.positions.y.data,
            simulation_.current_state.positions.z.data,
            topology.bending.first.data,
            topology.bending.second.data,
            bending_rest_lengths_,
            bending_width_,
            strain_range_,
            visualization_cuda::SegmentStyle::bending,
            bending_segments_.mapped_buffers[frame_slot_index]);
        for (std::uint32_t sample = 0u; sample < 3u; ++sample) {
            visualization_cuda::launch_wind_arrow(
                stream,
                simulation_.options.width * static_cast<float>(sample + 1u) * 0.25F,
                0.18F,
                0.0F,
                0.0F,
                static_cast<float>(simulation_.metrics.sampled_load_accelerations[sample]),
                load_scale_,
                load_width_,
                visualization_cuda::SegmentStyle::estimated_wind,
                static_cast<std::byte*>(load_segments_.mapped_buffers[frame_slot_index]) + static_cast<std::size_t>(sample) * 3u * segment_bytes);
        }
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error(std::format("cloth visualization kernel launch failed: {}", cudaGetErrorString(status)));
        simulation_.context.synchronize();
        stretch_segments_.slot_revisions[frame_slot_index] = content_revision_;
        bending_segments_.slot_revisions[frame_slot_index] = content_revision_;
        load_segments_.slot_revisions[frame_slot_index] = content_revision_;
    }

} // namespace xayah::cloth::examples::forward::project

extern "C" XAYAH_CLOTH_FORWARD_PLUGIN_EXPORT auto spectra_scene_plugin_v21() -> decltype(xayah::spectra::plugin::export_plugin<xayah::cloth::examples::forward::project::Project>()) {
    return xayah::spectra::plugin::export_plugin<xayah::cloth::examples::forward::project::Project>();
}
