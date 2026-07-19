#if defined(_WIN32)
#define XAYAH_SMOKE_FORWARD_PLUGIN_EXPORT __declspec(dllexport)
#else
#define XAYAH_SMOKE_FORWARD_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime_api.h>

import std;
import xayah.examples.smoke.forward;
import xayah.smoke.data;
import xayah.spectra.plugin;

namespace xayah::smoke::examples::forward::project {

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

        struct SegmentInstance {
            float sx;
            float sy;
            float sz;
            float width;
            float ex;
            float ey;
            float ez;
            std::uint32_t flags;
            float r;
            float g;
            float b;
            float a;
        };

        static_assert(sizeof(SegmentInstance) == segment_bytes);

        plugin::Camera overview_camera() {
            const std::array<float, 3u> position{1.90F, 1.20F, 3.10F};
            const std::array<float, 3u> target{0.50F, 0.70F, 0.50F};
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
                .vertical_fov_degrees = 35.0F,
                .near_plane = 0.01F,
                .far_plane = 20.0F,
            };
        }

        std::array<SegmentInstance, 3u> emitter_arrow(const Vector3 origin, const Vector3 acceleration, const float scale, const float width, const std::array<float, 4u> color) {
            const Vector3 vector{scale * acceleration.x, scale * acceleration.y, scale * acceleration.z};
            const float length = std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
            const Vector3 direction{vector.x / length, vector.y / length, vector.z / length};
            const Vector3 end{origin.x + vector.x, origin.y + vector.y, origin.z + vector.z};
            const Vector3 side{-direction.z / std::sqrt(direction.x * direction.x + direction.z * direction.z), 0.0F, direction.x / std::sqrt(direction.x * direction.x + direction.z * direction.z)};
            const float head_length = 0.28F * length;
            const float head_width = 0.16F * length;
            const Vector3 base{end.x - head_length * direction.x, end.y - head_length * direction.y, end.z - head_length * direction.z};
            return {{
                {.sx = origin.x, .sy = origin.y, .sz = origin.z, .width = width, .ex = end.x, .ey = end.y, .ez = end.z, .flags = 0u, .r = color[0], .g = color[1], .b = color[2], .a = color[3]},
                {.sx = end.x, .sy = end.y, .sz = end.z, .width = width, .ex = base.x + head_width * side.x, .ey = base.y, .ez = base.z + head_width * side.z, .flags = 0u, .r = color[0], .g = color[1], .b = color[2], .a = color[3]},
                {.sx = end.x, .sy = end.y, .sz = end.z, .width = width, .ex = base.x - head_width * side.x, .ey = base.y, .ez = base.z - head_width * side.z, .flags = 0u, .r = color[0], .g = color[1], .b = color[2], .a = color[3]},
            }};
        }

    } // namespace

    struct Project final {
        std::uint64_t revision{1u};

        explicit Project(std::shared_ptr<plugin::HostServices> host_services);
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
        ExternalBuffer density_{};
        ExternalBuffer emitters_{};
        std::uint64_t content_revision_{1u};
        std::uint32_t current_frame_slot_{};
        bool update_running_{};
        bool reset_pending_{};
        bool show_emitters_{true};
        float density_scale_{10.0F};
        float emitter_width_{3.0F};
        float emitter_scale_{0.04F};
    };

    Project::Project(std::shared_ptr<plugin::HostServices> host_services) {
        const std::uint64_t cell_count = static_cast<std::uint64_t>(simulation_.options.resolution[0]) * simulation_.options.resolution[1] * simulation_.options.resolution[2];
        density_.create(host_services, plugin::GpuBufferKindVolumeChannel, cell_count * sizeof(float));
        emitters_.create(host_services, plugin::GpuBufferKindViewportSegmentSet, 6u * segment_bytes);
        for (std::size_t frame_slot = 0u; frame_slot < density_.mapped_buffers.size(); ++frame_slot) write_visualization(static_cast<std::uint32_t>(frame_slot));
    }

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition = [] {
            plugin::PluginDefinition<Project> value{
                .id = "xayah.examples.smoke.forward",
                .title = "CUDA 3D Double-Jet Forward Smoke",
                .open_action_label = "Open CUDA 3D Double-Jet Forward Smoke",
                .sections = {{.id = "simulation", .label = "Simulation"}, {.id = "display", .label = "Display"}},
                .actions = {plugin::action<Project>("reset", "Reset Simulation", "Restore the empty state and physical step zero on the next safe frame-slot update.", "simulation", [](Project& project) { project.reset_pending_ = true; })},
                .settings = {
                    plugin::float_setting<Project>("density_scale", "Density Scale", 10.0F, "display", 0.05F, 20.0F, 0.05F, [](Project& project, const float value) {
                        project.density_scale_ = value;
                        ++project.revision;
                    }),
                    plugin::toggle<Project>("show_emitters", "Show Emitters", true, "display", [](Project& project, const bool value) {
                        project.show_emitters_ = value;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("emitter_width", "Emitter Width", 3.0F, "display", 0.25F, 8.0F, 0.25F, [](Project& project, const float value) {
                        project.emitter_width_ = value;
                        ++project.content_revision_;
                        ++project.revision;
                    }),
                    plugin::float_setting<Project>("emitter_scale", "Emitter Scale", 0.04F, "display", 0.01F, 0.12F, 0.005F, [](Project& project, const float value) {
                        project.emitter_scale_ = value;
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
        if (density_.slot_revisions[update.frame_slot_index] != content_revision_ || emitters_.slot_revisions[update.frame_slot_index] != content_revision_) write_visualization(update.frame_slot_index);
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        scene.set_document(plugin::Document{
            .update = {.enabled = true, .initial_running = false, .step_delta_seconds = simulation_.options.time_step},
            .navigation_target = {
                .revision = 1u,
                .focus = {0.50F, 0.70F, 0.50F},
                .bounds_minimum = {-0.20F, -0.10F, -0.20F},
                .bounds_maximum = {1.20F, 1.60F, 1.20F},
                .navigation_up = {0.0F, 1.0F, 0.0F},
            },
            .active_camera_name = "Overview",
            .materials = {{
                .name = "Smoke",
                .base_color = {0.72F, 0.76F, 0.80F, 0.92F},
                .density = {.channel_name = "density", .component = 0u, .scale = density_scale_, .bias = 0.0F, .enabled = true},
            }},
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo) const {
        plugin::Document document{
            .cameras = {overview_camera()},
            .volumes = {{
                .name = "Double-Jet Smoke",
                .dimensions = simulation_.options.resolution,
                .origin = {0.0F, 0.0F, 0.0F},
                .voxel_size = {simulation_.options.cell_size, simulation_.options.cell_size, simulation_.options.cell_size},
                .channels = {{.name = "density", .format = plugin::VolumeChannelFormat::float32, .buffer_id = density_.allocation.resource_id, .device_pointer = reinterpret_cast<std::uintptr_t>(density_.mapped_buffers[current_frame_slot_]), .ready_event = reinterpret_cast<std::uintptr_t>(density_.ready_events[current_frame_slot_]), .source_byte_size = density_.allocation.byte_size}},
                .material_name = "Smoke",
            }},
        };
        if (show_emitters_) document.viewport_segment_sets.push_back({.name = "Emitters", .owner_name = "Overview", .segment_count = 6u, .buffer_id = emitters_.allocation.resource_id, .source_byte_size = emitters_.allocation.byte_size, .width = emitter_width_});
        scene.set_document(std::move(document));
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        const ForwardSimulationMetrics& metrics = simulation_.metrics;
        controls
            .phase(update_running_ ? "Forward Simulation" : "Paused")
            .headline("128 x 192 x 128 double-jet forward smoke")
            .message("Bottom playback advances the physical simulation directly. Two pulsed Gaussian emitters collide near the center and vorticity confinement exposes the resulting shear structures.")
            .metric("grid", "Grid", std::format("{} x {} x {}", simulation_.options.resolution[0], simulation_.options.resolution[1], simulation_.options.resolution[2]), "simulation")
            .metric("step", "Physical Step", std::to_string(metrics.step), "simulation")
            .metric("time", "Physical Time", std::format("{:.3f} s", metrics.physical_time), "simulation")
            .metric("pressure", "Pressure RBGS", std::format("{} iterations", simulation_.options.pressure_iterations), "simulation")
            .metric("vorticity", "Vorticity Confinement", std::format("{:.2f}", simulation_.options.vorticity_confinement), "simulation")
            .metric("density_mass", "Density Mass", std::format("{:.7f}", metrics.density_mass), "simulation")
            .metric("density_max", "Density Maximum", std::format("{:.6f}", metrics.density_maximum), "simulation")
            .metric("temperature_max", "Temperature Maximum", std::format("{:.6f}", metrics.temperature_maximum), "simulation")
            .metric("velocity_max", "Maximum Velocity", std::format("{:.6f}", metrics.maximum_velocity), "simulation")
            .metric("cfl", "CFL", std::format("{:.5f}", metrics.cfl), "simulation")
            .metric("pre_divergence", "Pre-Projection Divergence RMS", std::format("{:.7e}", metrics.pre_projection_divergence_rms), "simulation")
            .metric("post_divergence", "Post-Projection Divergence RMS", std::format("{:.7e}", metrics.post_projection_divergence_rms), "simulation")
            .metric("divergence_ratio", "Divergence Ratio", std::format("{:.5f}", metrics.divergence_ratio), "simulation")
            .metric("step_time", "Forward Step", std::format("{:.3f} ms", metrics.step_milliseconds), "simulation")
            .metric("average_step_time", "Average Step", std::format("{:.3f} ms", metrics.average_step_milliseconds), "simulation")
            .metric("slot", "Frame Slot", std::to_string(current_frame_slot_), "display")
            .setting("density_scale", std::format("{}", density_scale_))
            .setting("show_emitters", show_emitters_ ? "true" : "false")
            .setting("emitter_width", std::format("{}", emitter_width_))
            .setting("emitter_scale", std::format("{}", emitter_scale_))
            .enable("reset");
    }

    void Project::write_visualization(const std::uint32_t frame_slot_index) {
        const cudaStream_t stream = static_cast<cudaStream_t>(simulation_.context.resource.native_stream);
        simulation_.context.resource.copy_device(density_.mapped_buffers[frame_slot_index], simulation_.current_state.density.values.data, density_.allocation.byte_size);
        const std::array<SegmentInstance, 3u> left = emitter_arrow(simulation_.options.left_source_center, simulation_.options.left_acceleration, emitter_scale_, emitter_width_, {0.10F, 0.84F, 0.96F, 1.00F});
        const std::array<SegmentInstance, 3u> right = emitter_arrow(simulation_.options.right_source_center, simulation_.options.right_acceleration, emitter_scale_, emitter_width_, {1.00F, 0.48F, 0.08F, 1.00F});
        std::array<SegmentInstance, 6u> segments{};
        std::ranges::copy(left, segments.begin());
        std::ranges::copy(right, segments.begin() + 3u);
        simulation_.context.resource.copy_from_host(emitters_.mapped_buffers[frame_slot_index], segments.data(), sizeof(segments));
        density_.record_ready(frame_slot_index, stream);
        emitters_.record_ready(frame_slot_index, stream);
        simulation_.context.resource.synchronize();
        density_.slot_revisions[frame_slot_index] = content_revision_;
        emitters_.slot_revisions[frame_slot_index] = content_revision_;
    }

} // namespace xayah::smoke::examples::forward::project

extern "C" XAYAH_SMOKE_FORWARD_PLUGIN_EXPORT auto spectra_scene_plugin_v21() -> decltype(xayah::spectra::plugin::export_plugin<xayah::smoke::examples::forward::project::Project>()) {
    return xayah::spectra::plugin::export_plugin<xayah::smoke::examples::forward::project::Project>();
}
