#if defined(_WIN32)
#include <windows.h>
#define XAYAH_STRETCH_STIFFNESS_INVERSE_PLUGIN_EXPORT __declspec(dllexport)
#else
#include <unistd.h>
#define XAYAH_STRETCH_STIFFNESS_INVERSE_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#include <cuda_runtime_api.h>

#include "project.h"

import std;
import xayah.cloth.data;
import xayah.examples.cloth.stretch_stiffness_inverse;
import xayah.cloth.model;
import xayah.spectra.plugin;
import xayah.cloth.runtime;

namespace xayah::cloth::examples::stretch_stiffness_inverse::project {

    namespace plugin = spectra::plugin;

    namespace {

        constexpr std::uint64_t segment_bytes = 48u;

        struct ProjectOptions {
            std::uint32_t rows{16u};
            std::uint32_t columns{16u};
            float width{2.0F};
            float height{2.0F};
            float mass{0.05F};
            float target_stretch_stiffness{400.0F};
            float initial_stretch_stiffness{100.0F};
            float stretch_damping{1.0F};
            float bending_stiffness{5.0F};
            float bending_damping{0.1F};
            float gravity_y{-9.81F};
            float time_step{1.0F / 240.0F};
            std::uint32_t trajectory_steps{120u};
            float adam_learning_rate{0.05F};
            std::uint32_t iterations_per_update{1u};
        };

        void close_imported_handle(plugin::GpuBufferSlotAllocation& slot) noexcept {
#if defined(_WIN32)
            if (slot.handle_kind == plugin::GpuResourceHandleKind::OpaqueWin32 && slot.handle != 0u && CloseHandle(reinterpret_cast<HANDLE>(slot.handle)) == 0) std::terminate();
#else
            if (slot.handle_kind == plugin::GpuResourceHandleKind::OpaqueFileDescriptor && slot.handle != 0u && close(static_cast<int>(slot.handle)) != 0) std::terminate();
#endif
            slot.handle = 0u;
        }

        class ExternalGpuBuffer final {
        public:
            ExternalGpuBuffer() = default;
            ExternalGpuBuffer(const ExternalGpuBuffer&) = delete;
            ExternalGpuBuffer(ExternalGpuBuffer&&) = delete;
            ExternalGpuBuffer& operator=(const ExternalGpuBuffer&) = delete;
            ExternalGpuBuffer& operator=(ExternalGpuBuffer&&) = delete;
            ~ExternalGpuBuffer() noexcept;

            void create(std::shared_ptr<plugin::HostServices> host_services, std::uint64_t byte_size);

            template <typename Element>
            [[nodiscard]] Element* mapped_as(const std::uint32_t frame_slot_index) const noexcept {
                return static_cast<Element*>(this->mapped_buffers_[frame_slot_index]);
            }

            [[nodiscard]] std::uint64_t resource_id() const noexcept;
            [[nodiscard]] std::uint64_t byte_size() const noexcept;
            [[nodiscard]] std::uint32_t frame_slot_count() const noexcept;
            [[nodiscard]] std::uint64_t slot_revision(std::uint32_t frame_slot_index) const noexcept;
            void set_slot_revision(std::uint32_t frame_slot_index, std::uint64_t revision) noexcept;

        private:
            std::shared_ptr<plugin::HostServices> host_services_{};
            plugin::GpuBufferAllocation allocation_{};
            std::vector<cudaExternalMemory_t> external_memories_{};
            std::vector<void*> mapped_buffers_{};
            std::vector<std::uint64_t> slot_revisions_{};
            std::uint64_t byte_size_{};
        };

        class CudaEventTimer final {
        public:
            CudaEventTimer();
            CudaEventTimer(const CudaEventTimer&) = delete;
            CudaEventTimer(CudaEventTimer&&) = delete;
            CudaEventTimer& operator=(const CudaEventTimer&) = delete;
            CudaEventTimer& operator=(CudaEventTimer&&) = delete;
            ~CudaEventTimer() noexcept;

            void begin(cudaStream_t stream);
            [[nodiscard]] float finish(cudaStream_t stream);

        private:
            cudaEvent_t begin_{};
            cudaEvent_t end_{};
        };

        [[nodiscard]] Configuration make_configuration(const ProjectOptions& options);
        [[nodiscard]] plugin::Camera overview_camera(float width, float height);

    } // namespace

    class Project final {
    public:
        struct State;

        Project(const Project&) = delete;
        Project(Project&&) noexcept;
        Project& operator=(const Project&) = delete;
        Project& operator=(Project&&) noexcept;
        ~Project() noexcept;

        [[nodiscard]] static const plugin::PluginDefinition<Project>& plugin();
        [[nodiscard]] static Project open(plugin::OpenContext context);

        void update(const plugin::UpdateInfo& update);
        [[nodiscard]] std::uint64_t revision() const;
        void write_document(plugin::SceneBuilder& scene) const;
        void write_frame(plugin::SceneBuilder& scene, plugin::FrameInfo frame) const;
        void write_controls(plugin::ControlBuilder& controls) const;

        void reset_optimization();
        void set_trajectory_frame(std::uint64_t value);
        void set_show_target(bool value);
        void set_show_estimate(bool value);
        void set_show_bending(bool value);
        void set_stretch_width(float value);
        void set_bending_width(float value);
        void set_strain_range(float value);

    private:
        explicit Project(std::unique_ptr<State> state);

        std::unique_ptr<State> state_;
    };

    struct Project::State final {
        State(ProjectOptions options, Configuration configuration, std::shared_ptr<plugin::HostServices> host_services);

        ProjectOptions options{};
        std::shared_ptr<plugin::HostServices> host_services{};
        StretchStiffnessInverseTask task;
        Parameters visualization_parameters;
        ExternalGpuBuffer target_stretch_segments{};
        ExternalGpuBuffer estimated_stretch_segments{};
        ExternalGpuBuffer estimated_bending_segments{};
        std::uint64_t scene_revision{1u};
        std::uint64_t content_revision{1u};
        float cuda_update_milliseconds{};
        std::uint32_t current_frame_slot{};
        std::uint32_t trajectory_step;
        bool update_running{};
        bool reset_pending{};
        bool show_target{true};
        bool show_estimate{true};
        bool show_bending{};
        float stretch_width{2.0F};
        float bending_width{1.0F};
        float strain_range{0.10F};
    };

    namespace {

        ExternalGpuBuffer::~ExternalGpuBuffer() noexcept {
            for (void* const mapped_buffer : this->mapped_buffers_)
                if (mapped_buffer != nullptr && cudaFree(mapped_buffer) != cudaSuccess) std::terminate();
            for (const cudaExternalMemory_t external_memory : this->external_memories_)
                if (external_memory != nullptr && cudaDestroyExternalMemory(external_memory) != cudaSuccess) std::terminate();
            if (this->allocation_.resource_id != 0u) {
                try {
                    this->host_services_->release_gpu_buffer(this->allocation_.resource_id);
                } catch (...) {
                    std::terminate();
                }
            }
        }

        void ExternalGpuBuffer::create(std::shared_ptr<plugin::HostServices> host_services, const std::uint64_t byte_size) {
            plugin::GpuBufferAllocation allocation = host_services->request_gpu_buffer(plugin::GpuBufferKindViewportSegmentSet, byte_size);
            std::vector<cudaExternalMemory_t> external_memories{};
            std::vector<void*> mapped_buffers{};
            external_memories.reserve(allocation.slots.size());
            mapped_buffers.reserve(allocation.slots.size());
            try {
                for (plugin::GpuBufferSlotAllocation& slot : allocation.slots) {
                    cudaExternalMemoryHandleDesc memory_descriptor{};
                    memory_descriptor.size = allocation.byte_size;
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
                    buffer_descriptor.size = allocation.byte_size;
                    void* mapped_buffer{};
                    const cudaError_t mapping_status = cudaExternalMemoryGetMappedBuffer(&mapped_buffer, external_memory, &buffer_descriptor);
                    if (mapping_status != cudaSuccess) {
                        static_cast<void>(cudaDestroyExternalMemory(external_memory));
                        throw std::runtime_error(std::format("cudaExternalMemoryGetMappedBuffer failed: {}", cudaGetErrorString(mapping_status)));
                    }
                    external_memories.push_back(external_memory);
                    mapped_buffers.push_back(mapped_buffer);
                }
            } catch (...) {
                for (plugin::GpuBufferSlotAllocation& slot : allocation.slots)
                    if (slot.handle != 0u) close_imported_handle(slot);
                for (void* const mapped_buffer : mapped_buffers)
                    if (mapped_buffer != nullptr) static_cast<void>(cudaFree(mapped_buffer));
                for (const cudaExternalMemory_t external_memory : external_memories)
                    if (external_memory != nullptr) static_cast<void>(cudaDestroyExternalMemory(external_memory));
                host_services->release_gpu_buffer(allocation.resource_id);
                throw;
            }
            this->host_services_ = std::move(host_services);
            this->allocation_ = std::move(allocation);
            this->external_memories_ = std::move(external_memories);
            this->mapped_buffers_ = std::move(mapped_buffers);
            this->slot_revisions_.assign(this->mapped_buffers_.size(), 0u);
            this->byte_size_ = byte_size;
        }

        std::uint64_t ExternalGpuBuffer::resource_id() const noexcept {
            return this->allocation_.resource_id;
        }

        std::uint64_t ExternalGpuBuffer::byte_size() const noexcept {
            return this->byte_size_;
        }

        std::uint32_t ExternalGpuBuffer::frame_slot_count() const noexcept {
            return static_cast<std::uint32_t>(this->mapped_buffers_.size());
        }

        std::uint64_t ExternalGpuBuffer::slot_revision(const std::uint32_t frame_slot_index) const noexcept {
            return this->slot_revisions_[frame_slot_index];
        }

        void ExternalGpuBuffer::set_slot_revision(const std::uint32_t frame_slot_index, const std::uint64_t revision) noexcept {
            this->slot_revisions_[frame_slot_index] = revision;
        }

        CudaEventTimer::CudaEventTimer() {
            if (const cudaError_t status = cudaEventCreate(&this->begin_); status != cudaSuccess) throw std::runtime_error(std::format("cudaEventCreate failed: {}", cudaGetErrorString(status)));
            try {
                if (const cudaError_t status = cudaEventCreate(&this->end_); status != cudaSuccess) throw std::runtime_error(std::format("cudaEventCreate failed: {}", cudaGetErrorString(status)));
            } catch (...) {
                static_cast<void>(cudaEventDestroy(this->begin_));
                throw;
            }
        }

        CudaEventTimer::~CudaEventTimer() noexcept {
            if (cudaEventDestroy(this->begin_) != cudaSuccess) std::terminate();
            if (cudaEventDestroy(this->end_) != cudaSuccess) std::terminate();
        }

        void CudaEventTimer::begin(const cudaStream_t stream) {
            if (const cudaError_t status = cudaEventRecord(this->begin_, stream); status != cudaSuccess) throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorString(status)));
        }

        float CudaEventTimer::finish(const cudaStream_t stream) {
            if (const cudaError_t status = cudaEventRecord(this->end_, stream); status != cudaSuccess) throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorString(status)));
            if (const cudaError_t status = cudaEventSynchronize(this->end_); status != cudaSuccess) throw std::runtime_error(std::format("cudaEventSynchronize failed: {}", cudaGetErrorString(status)));
            float milliseconds{};
            if (const cudaError_t status = cudaEventElapsedTime(&milliseconds, this->begin_, this->end_); status != cudaSuccess) throw std::runtime_error(std::format("cudaEventElapsedTime failed: {}", cudaGetErrorString(status)));
            return milliseconds;
        }

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

        void write_visualization(Project::State& state, const std::uint32_t frame_slot_index) {
            ExecutionContext& context       = state.task.context();
            const Model& model              = state.task.model();
            const cudaStream_t stream       = static_cast<cudaStream_t>(context.resource().native_stream());
            const DeviceTopology& topology  = context.device_topology();
            const ::xayah::cloth::State& target_state    = state.task.target_state(state.trajectory_step);
            const ::xayah::cloth::State& estimated_state = state.task.estimated_state(state.trajectory_step);
            visualization_cuda::launch_segments(
                stream,
                static_cast<std::uint32_t>(model.topology().stretch_springs.size()),
                target_state.positions.x.data(),
                target_state.positions.y.data(),
                target_state.positions.z.data(),
                topology.stretch.first.data(),
                topology.stretch.second.data(),
                state.visualization_parameters.stretch_rest_lengths.data(),
                1.0F,
                state.strain_range,
                visualization_cuda::SegmentStyle::target,
                state.target_stretch_segments.mapped_as<void>(frame_slot_index));
            visualization_cuda::launch_segments(
                stream,
                static_cast<std::uint32_t>(model.topology().stretch_springs.size()),
                estimated_state.positions.x.data(),
                estimated_state.positions.y.data(),
                estimated_state.positions.z.data(),
                topology.stretch.first.data(),
                topology.stretch.second.data(),
                state.visualization_parameters.stretch_rest_lengths.data(),
                state.stretch_width,
                state.strain_range,
                visualization_cuda::SegmentStyle::estimate,
                state.estimated_stretch_segments.mapped_as<void>(frame_slot_index));
            visualization_cuda::launch_segments(
                stream,
                static_cast<std::uint32_t>(model.topology().bending_springs.size()),
                estimated_state.positions.x.data(),
                estimated_state.positions.y.data(),
                estimated_state.positions.z.data(),
                topology.bending.first.data(),
                topology.bending.second.data(),
                state.visualization_parameters.bending_rest_lengths.data(),
                state.bending_width,
                state.strain_range,
                visualization_cuda::SegmentStyle::bending,
                state.estimated_bending_segments.mapped_as<void>(frame_slot_index));
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error(std::format("cloth visualization kernel launch failed: {}", cudaGetErrorString(status)));
            state.target_stretch_segments.set_slot_revision(frame_slot_index, state.content_revision);
            state.estimated_stretch_segments.set_slot_revision(frame_slot_index, state.content_revision);
            state.estimated_bending_segments.set_slot_revision(frame_slot_index, state.content_revision);
        }

        void write_all_visualization_slots(Project::State& state) {
            for (std::uint32_t frame_slot = 0u; frame_slot < state.target_stretch_segments.frame_slot_count(); ++frame_slot) write_visualization(state, frame_slot);
            state.task.context().synchronize();
        }

        void reset_task(Project::State& state) {
            state.task.reset();
            state.reset_pending = false;
            ++state.content_revision;
        }

    } // namespace

    Project::State::State(ProjectOptions next_options, Configuration configuration, std::shared_ptr<plugin::HostServices> next_host_services)
        : options(next_options), host_services(std::move(next_host_services)), task(std::move(configuration), StretchStiffnessInverseOptions{
              .mass = options.mass,
              .target_stretch_stiffness = options.target_stretch_stiffness,
              .initial_stretch_stiffness = options.initial_stretch_stiffness,
              .stretch_damping = options.stretch_damping,
              .bending_stiffness = options.bending_stiffness,
              .bending_damping = options.bending_damping,
              .trajectory_steps = options.trajectory_steps,
              .adam_learning_rate = options.adam_learning_rate,
          }), visualization_parameters(task.model().make_parameters(task.context())), trajectory_step(options.trajectory_steps) {
        std::vector<float> stretch_rest_lengths(task.model().topology().stretch_springs.size());
        for (std::size_t spring = 0u; spring < stretch_rest_lengths.size(); ++spring) stretch_rest_lengths[spring] = task.model().topology().stretch_springs[spring].rest_length;
        task.context().upload(stretch_rest_lengths, visualization_parameters.stretch_rest_lengths);
        std::vector<float> bending_rest_lengths(task.model().topology().bending_springs.size());
        for (std::size_t spring = 0u; spring < bending_rest_lengths.size(); ++spring) bending_rest_lengths[spring] = task.model().topology().bending_springs[spring].rest_length;
        task.context().upload(bending_rest_lengths, visualization_parameters.bending_rest_lengths);
        target_stretch_segments.create(host_services, task.model().topology().stretch_springs.size() * segment_bytes);
        estimated_stretch_segments.create(host_services, task.model().topology().stretch_springs.size() * segment_bytes);
        estimated_bending_segments.create(host_services, task.model().topology().bending_springs.size() * segment_bytes);
        write_all_visualization_slots(*this);
    }

    Project::Project(std::unique_ptr<State> state) : state_(std::move(state)) {}

    Project::Project(Project&&) noexcept = default;

    Project& Project::operator=(Project&&) noexcept = default;

    Project::~Project() noexcept = default;

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition = [] {
            plugin::PluginDefinition<Project> value{
                .id = "xayah.examples.cloth.stretch_stiffness_inverse",
                .title = "CUDA Cloth Stiffness Inverse",
                .open_action_label = "Open CUDA Cloth Stiffness Inverse",
                .sections = {{.id = "inverse", .label = "Inverse"}, {.id = "physics", .label = "Physics"}, {.id = "display", .label = "Display"}},
                .open_options = {
                    plugin::unsigned_integer("rows", "Rows", 16u, "physics"),
                    plugin::unsigned_integer("columns", "Columns", 16u, "physics"),
                    plugin::float_value("width", "Width", 2.0F, "physics"),
                    plugin::float_value("height", "Height", 2.0F, "physics"),
                    plugin::float_value("mass", "Mass", 0.05F, "physics"),
                    plugin::float_value("stretch_damping", "Stretch Damping", 1.0F, "physics"),
                    plugin::float_value("bending_stiffness", "Bending Stiffness", 5.0F, "physics"),
                    plugin::float_value("bending_damping", "Bending Damping", 0.1F, "physics"),
                    plugin::float_value("gravity_y", "Gravity Y", -9.81F, "physics"),
                    plugin::float_value("time_step", "Time Step", 1.0F / 240.0F, "physics"),
                    plugin::float_value("target_stretch_stiffness", "Target Stretch Stiffness", 400.0F, "inverse"),
                    plugin::float_value("initial_stretch_stiffness", "Initial Stretch Stiffness", 100.0F, "inverse"),
                    plugin::unsigned_integer("trajectory_steps", "Trajectory Steps", 120u, "inverse"),
                    plugin::float_value("adam_learning_rate", "Adam Learning Rate", 0.05F, "inverse"),
                    plugin::unsigned_integer("iterations_per_update", "Iterations Per Update", 1u, "inverse"),
                },
                .actions = {plugin::action("reset", "Reset Optimization", "Restore the initial stiffness and Adam state on the next safe frame-slot update.", "inverse", &Project::reset_optimization)},
                .settings = {
                    plugin::unsigned_integer_setting("trajectory_frame", "Trajectory Frame", 120u, "display", 0u, 120u, 1u, &Project::set_trajectory_frame),
                    plugin::toggle("show_target", "Show Target", true, "display", &Project::set_show_target),
                    plugin::toggle("show_estimate", "Show Estimate", true, "display", &Project::set_show_estimate),
                    plugin::toggle("show_bending", "Show Bending", false, "display", &Project::set_show_bending),
                    plugin::float_setting("stretch_width", "Stretch Width", 2.0F, "display", 0.25F, 8.0F, 0.25F, &Project::set_stretch_width),
                    plugin::float_setting("bending_width", "Bending Width", 1.0F, "display", 0.25F, 8.0F, 0.25F, &Project::set_bending_width),
                    plugin::float_setting("strain_range", "Strain Range", 0.10F, "display", 0.01F, 0.50F, 0.01F, &Project::set_strain_range),
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
            else if (option.key == "target_stretch_stiffness") options.target_stretch_stiffness = std::stof(option.value);
            else if (option.key == "initial_stretch_stiffness") options.initial_stretch_stiffness = std::stof(option.value);
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
        return Project(std::make_unique<State>(options, std::move(configuration), std::move(context.host_services)));
    }

    void Project::update(const plugin::UpdateInfo& update) {
        this->state_->current_frame_slot = update.frame_slot_index;
        this->state_->update_running = update.update_running;
        CudaEventTimer timer{};
        const cudaStream_t stream = static_cast<cudaStream_t>(this->state_->task.context().resource().native_stream());
        timer.begin(stream);
        if (this->state_->reset_pending) reset_task(*this->state_);
        if (update.update_delta_seconds > 0.0) {
            for (std::uint32_t iteration = 0u; iteration < this->state_->options.iterations_per_update; ++iteration) this->state_->task.optimize_step();
            ++this->state_->content_revision;
        }
        if (this->state_->target_stretch_segments.slot_revision(update.frame_slot_index) != this->state_->content_revision || this->state_->estimated_stretch_segments.slot_revision(update.frame_slot_index) != this->state_->content_revision || this->state_->estimated_bending_segments.slot_revision(update.frame_slot_index) != this->state_->content_revision) write_visualization(*this->state_, update.frame_slot_index);
        this->state_->cuda_update_milliseconds = timer.finish(stream);
    }

    std::uint64_t Project::revision() const {
        return this->state_->scene_revision;
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        scene.set_document(plugin::Document{
            .update = {.enabled = true, .initial_running = false, .step_delta_seconds = 1.0 / 60.0},
            .navigation_target = {
                .revision = 1u,
                .focus = {this->state_->options.width * 0.5F, -this->state_->options.height * 0.45F, this->state_->options.height * 0.5F},
                .bounds_minimum = {-this->state_->options.width * 0.1F, -this->state_->options.height * 1.25F, -this->state_->options.height * 0.1F},
                .bounds_maximum = {this->state_->options.width * 1.1F, this->state_->options.height * 0.25F, this->state_->options.height * 1.1F},
                .navigation_up = {0.0F, 1.0F, 0.0F},
            },
            .active_camera_name = "Overview",
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo) const {
        plugin::Document document{.cameras = {overview_camera(this->state_->options.width, this->state_->options.height)}};
        if (this->state_->show_target) document.viewport_segment_sets.push_back({.name = "Target Stretch Springs", .owner_name = "Overview", .segment_count = this->state_->task.model().topology().stretch_springs.size(), .buffer_id = this->state_->target_stretch_segments.resource_id(), .source_byte_size = this->state_->target_stretch_segments.byte_size(), .width = 1.0F});
        if (this->state_->show_estimate) document.viewport_segment_sets.push_back({.name = "Estimated Stretch Springs", .owner_name = "Overview", .segment_count = this->state_->task.model().topology().stretch_springs.size(), .buffer_id = this->state_->estimated_stretch_segments.resource_id(), .source_byte_size = this->state_->estimated_stretch_segments.byte_size(), .width = this->state_->stretch_width});
        if (this->state_->show_bending) document.viewport_segment_sets.push_back({.name = "Estimated Bending Springs", .owner_name = "Overview", .segment_count = this->state_->task.model().topology().bending_springs.size(), .buffer_id = this->state_->estimated_bending_segments.resource_id(), .source_byte_size = this->state_->estimated_bending_segments.byte_size(), .width = this->state_->bending_width});
        scene.set_document(std::move(document));
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        const StretchStiffnessInverseMetrics& metrics = this->state_->task.metrics();
        const double stiffness_error = std::abs(static_cast<double>(metrics.stretch_stiffness) - this->state_->options.target_stretch_stiffness) / this->state_->options.target_stretch_stiffness;
        controls
            .phase(this->state_->update_running ? "Optimizing" : "Paused")
            .headline("Differentiable stretch-stiffness inversion")
            .message("The full target trajectory, loss seeds, VJP, and segment instances remain on the GPU.")
            .metric("iteration", "Iteration", std::to_string(metrics.iteration), "inverse")
            .metric("stiffness", "Estimated Stiffness", std::format("{:.6f}", metrics.stretch_stiffness), "inverse")
            .metric("target", "Target Stiffness", std::format("{:.6f}", this->state_->options.target_stretch_stiffness), "inverse")
            .metric("stiffness_error", "Relative Stiffness Error", std::format("{:.3e}", stiffness_error), "inverse")
            .metric("loss", "Trajectory Loss", std::format("{:.9e}", metrics.loss), "inverse")
            .metric("loss_ratio", "Final / Initial Loss", std::format("{:.3e}", metrics.loss / metrics.initial_loss), "inverse")
            .metric("duration", "Trajectory Duration", std::format("{:.4f} s", static_cast<double>(this->state_->options.trajectory_steps) * this->state_->options.time_step), "inverse")
            .metric("grid", "Grid", std::format("{} x {}", this->state_->options.rows, this->state_->options.columns), "physics")
            .metric("cuda", "CUDA Optimization", std::format("{:.3f} ms", this->state_->cuda_update_milliseconds), "inverse")
            .metric("trajectory_frame", "Trajectory Frame", std::format("{} / {}", this->state_->trajectory_step, this->state_->options.trajectory_steps), "display")
            .metric("physical_time", "Physical Time", std::format("{:.4f} s", static_cast<double>(this->state_->trajectory_step) * this->state_->options.time_step), "display")
            .metric("slot", "Frame Slot", std::to_string(this->state_->current_frame_slot), "display")
            .unsigned_setting("trajectory_frame", this->state_->trajectory_step, 0u, this->state_->options.trajectory_steps, 1u)
            .setting("show_target", this->state_->show_target ? "true" : "false")
            .setting("show_estimate", this->state_->show_estimate ? "true" : "false")
            .setting("show_bending", this->state_->show_bending ? "true" : "false")
            .setting("stretch_width", std::format("{}", this->state_->stretch_width))
            .setting("bending_width", std::format("{}", this->state_->bending_width))
            .setting("strain_range", std::format("{}", this->state_->strain_range))
            .enable("reset");
    }

    void Project::reset_optimization() {
        this->state_->reset_pending = true;
    }

    void Project::set_trajectory_frame(const std::uint64_t value) {
        this->state_->trajectory_step = static_cast<std::uint32_t>(value);
        ++this->state_->content_revision;
    }

    void Project::set_show_target(const bool value) {
        this->state_->show_target = value;
        ++this->state_->scene_revision;
    }

    void Project::set_show_estimate(const bool value) {
        this->state_->show_estimate = value;
        ++this->state_->scene_revision;
    }

    void Project::set_show_bending(const bool value) {
        this->state_->show_bending = value;
        ++this->state_->scene_revision;
    }

    void Project::set_stretch_width(const float value) {
        this->state_->stretch_width = value;
        ++this->state_->content_revision;
    }

    void Project::set_bending_width(const float value) {
        this->state_->bending_width = value;
        ++this->state_->content_revision;
    }

    void Project::set_strain_range(const float value) {
        this->state_->strain_range = value;
        ++this->state_->content_revision;
    }

} // namespace xayah::cloth::examples::stretch_stiffness_inverse::project

extern "C" XAYAH_STRETCH_STIFFNESS_INVERSE_PLUGIN_EXPORT auto spectra_scene_plugin_v21() -> decltype(xayah::spectra::plugin::export_plugin<xayah::cloth::examples::stretch_stiffness_inverse::project::Project>()) {
    return xayah::spectra::plugin::export_plugin<xayah::cloth::examples::stretch_stiffness_inverse::project::Project>();
}
