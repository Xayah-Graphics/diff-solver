export module xayah.spectra.plugin;

import std;

export namespace xayah::spectra::plugin {

    enum class GpuResourceHandleKind : std::uint32_t {
        OpaqueWin32 = 0u,
        OpaqueFileDescriptor = 1u,
    };

    struct GpuBufferSlotAllocation {
        GpuResourceHandleKind handle_kind{GpuResourceHandleKind::OpaqueWin32};
        std::uintptr_t handle{};
    };

    struct GpuBufferAllocation {
        std::uint64_t resource_id{};
        std::uint64_t byte_size{};
        std::vector<GpuBufferSlotAllocation> slots{};
    };

    inline constexpr std::uint32_t GpuBufferKindViewportSegmentSet = 3u;

    struct HostServices {
        std::move_only_function<GpuBufferAllocation(std::uint32_t, std::uint64_t)> request_gpu_buffer{};
        std::move_only_function<void(std::uint64_t)> release_gpu_buffer{};
    };

    struct Option {
        std::string key{};
        std::string value{};
    };

    enum class OptionKind : std::uint32_t {
        Bool = 4u,
        Float = 5u,
        UnsignedInteger = 6u,
    };

    struct Section {
        std::string id{};
        std::string label{};
    };

    struct OptionSchema {
        std::string key{};
        std::string label{};
        std::string description{};
        OptionKind kind{OptionKind::Float};
        std::string default_value{};
        std::string section_id{};
        bool slider{};
        float numeric_min{};
        float numeric_max{};
        float numeric_step{};
        std::uint64_t unsigned_min{};
        std::uint64_t unsigned_max{};
        std::uint64_t unsigned_step{};
    };

    struct Action {
        std::string id{};
        std::string label{};
        std::string description{};
        std::string section_id{};
    };

    struct UpdateInfo {
        double wall_delta_seconds{};
        double update_delta_seconds{};
        double timeline_time_seconds{};
        std::uint64_t timeline_frame_index{};
        std::uint32_t frame_slot_index{};
        std::uint32_t frame_slot_count{};
        bool update_running{};
    };

    struct FrameInfo {
        double time_seconds{};
        std::uint64_t frame_index{};
    };

    struct UpdateDescriptor {
        bool enabled{};
        bool initial_running{};
        double step_delta_seconds{};
    };

    struct NavigationTarget {
        std::uint64_t revision{1u};
        std::array<float, 3u> focus{};
        std::array<float, 3u> bounds_minimum{};
        std::array<float, 3u> bounds_maximum{};
        std::array<float, 3u> navigation_up{0.0F, 1.0F, 0.0F};
    };

    struct Camera {
        std::string name{};
        std::array<float, 3u> position{};
        std::array<float, 3u> right{1.0F, 0.0F, 0.0F};
        std::array<float, 3u> down{0.0F, -1.0F, 0.0F};
        std::array<float, 3u> forward{0.0F, 0.0F, -1.0F};
        float vertical_fov_degrees{45.0F};
        float near_plane{0.01F};
        float far_plane{100.0F};
    };

    struct ViewportSegmentSet {
        std::string name{};
        std::string owner_name{};
        std::uint64_t segment_count{};
        std::uint64_t buffer_id{};
        std::uint64_t source_byte_size{};
        float width{1.0F};
        bool overlay{};
    };

    struct Document {
        UpdateDescriptor update{};
        NavigationTarget navigation_target{};
        std::string active_camera_name{};
        std::vector<Camera> cameras{};
        std::vector<ViewportSegmentSet> viewport_segment_sets{};
    };

    class SceneBuilder {
    public:
        void set_document(Document document) {
            this->document_ = std::move(document);
        }

        [[nodiscard]] Document take_document() {
            return std::move(this->document_);
        }

    private:
        Document document_{};
    };

    struct Metric {
        std::string key{};
        std::string label{};
        std::string value{};
        std::string section_id{};
    };

    struct SettingState {
        std::string key{};
        std::string value{};
        bool has_unsigned_range{};
        std::uint64_t unsigned_min{};
        std::uint64_t unsigned_max{};
        std::uint64_t unsigned_step{};
    };

    struct ControlState {
        std::string phase{};
        std::string headline{};
        std::string detail{};
        std::vector<Metric> metrics{};
        std::vector<std::string> enabled_actions{};
        std::vector<SettingState> settings{};
    };

    class ControlBuilder {
    public:
        ControlBuilder& phase(std::string value) {
            this->state_.phase = std::move(value);
            return *this;
        }

        ControlBuilder& headline(std::string value) {
            this->state_.headline = std::move(value);
            return *this;
        }

        ControlBuilder& message(std::string value) {
            this->state_.detail = std::move(value);
            return *this;
        }

        ControlBuilder& metric(std::string key, std::string label, std::string value, std::string section_id = {}) {
            this->state_.metrics.push_back(Metric{.key = std::move(key), .label = std::move(label), .value = std::move(value), .section_id = std::move(section_id)});
            return *this;
        }

        ControlBuilder& enable(std::string action_id) {
            this->state_.enabled_actions.push_back(std::move(action_id));
            return *this;
        }

        ControlBuilder& setting(std::string key, std::string value) {
            this->state_.settings.push_back(SettingState{.key = std::move(key), .value = std::move(value)});
            return *this;
        }

        ControlBuilder& unsigned_setting(std::string key, const std::uint64_t value, const std::uint64_t minimum, const std::uint64_t maximum, const std::uint64_t step) {
            this->state_.settings.push_back(SettingState{.key = std::move(key), .value = std::to_string(value), .has_unsigned_range = true, .unsigned_min = minimum, .unsigned_max = maximum, .unsigned_step = step});
            return *this;
        }

        [[nodiscard]] ControlState take_state() {
            return std::move(this->state_);
        }

    private:
        ControlState state_{};
    };

    struct OpenContext {
        std::vector<Option> options{};
        std::shared_ptr<HostServices> host_services{};
    };

    template <typename Project>
    struct ActionBinding {
        Action schema{};
        std::function<void(Project&)> invoke{};
    };

    template <typename Project>
    struct SettingBinding {
        OptionSchema schema{};
        std::function<void(Project&, std::string_view)> update{};
    };

    template <typename Project>
    struct PluginDefinition {
        std::string id{};
        std::string title{};
        std::string open_action_label{};
        std::vector<Section> sections{};
        std::vector<OptionSchema> open_options{};
        std::vector<ActionBinding<Project>> actions{};
        std::vector<SettingBinding<Project>> settings{};
    };

    inline OptionSchema unsigned_integer(std::string key, std::string label, const std::uint64_t default_value, std::string section_id) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::UnsignedInteger, .default_value = std::to_string(default_value), .section_id = std::move(section_id)};
    }

    inline OptionSchema float_value(std::string key, std::string label, const float default_value, std::string section_id) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Float, .default_value = std::format("{}", default_value), .section_id = std::move(section_id)};
    }

    template <typename Project, std::invocable<Project&> Invoke>
    ActionBinding<Project> action(std::string id, std::string label, std::string description, std::string section_id, Invoke invoke) {
        return ActionBinding<Project>{
            .schema = Action{.id = std::move(id), .label = std::move(label), .description = std::move(description), .section_id = std::move(section_id)},
            .invoke = std::move(invoke),
        };
    }

    template <typename Project, std::invocable<Project&, bool> Update>
    SettingBinding<Project> toggle(std::string key, std::string label, const bool default_value, std::string section_id, Update update) {
        return SettingBinding<Project>{
            .schema = OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Bool, .default_value = default_value ? "true" : "false", .section_id = std::move(section_id)},
            .update = [update = std::move(update)](Project& project, const std::string_view value) { std::invoke(update, project, value == "true"); },
        };
    }

    template <typename Project, std::invocable<Project&, float> Update>
    SettingBinding<Project> float_setting(std::string key, std::string label, const float default_value, std::string section_id, const float minimum, const float maximum, const float step, Update update) {
        return SettingBinding<Project>{
            .schema = OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Float, .default_value = std::format("{}", default_value), .section_id = std::move(section_id), .slider = true, .numeric_min = minimum, .numeric_max = maximum, .numeric_step = step},
            .update = [update = std::move(update)](Project& project, const std::string_view value) { std::invoke(update, project, std::stof(std::string{value})); },
        };
    }

    template <typename Project, std::invocable<Project&, std::uint64_t> Update>
    SettingBinding<Project> unsigned_integer_setting(std::string key, std::string label, const std::uint64_t default_value, std::string section_id, const std::uint64_t minimum, const std::uint64_t maximum, const std::uint64_t step, Update update) {
        return SettingBinding<Project>{
            .schema = OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::UnsignedInteger, .default_value = std::to_string(default_value), .section_id = std::move(section_id), .slider = true, .unsigned_min = minimum, .unsigned_max = maximum, .unsigned_step = step},
            .update = [update = std::move(update)](Project& project, const std::string_view value) { std::invoke(update, project, std::stoull(std::string{value})); },
        };
    }

} // namespace xayah::spectra::plugin

namespace xayah::spectra::plugin {

    constexpr std::uint32_t plugin_abi_version = 21u;
    thread_local std::string plugin_export_error{};
    typedef void SpectraSceneInstance;
    typedef std::uint32_t SpectraSceneResult;
    constexpr std::uint32_t ResultOk = 0u;
    constexpr std::uint32_t ResultError = 1u;

    struct RawSpan {
        const void* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneOption {
        const char* key{};
        const char* value{};
    };

    struct SpectraSceneOptionSpan {
        const SpectraSceneOption* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlSection {
        const char* id{};
        const char* label{};
    };

    struct SpectraSceneControlOptionSchema {
        const char* key{};
        const char* label{};
        const char* description{};
        std::uint32_t kind{};
        std::uint32_t required{};
        const char* default_value{};
        const char* section_id{};
        RawSpan choices{};
        std::uint32_t presentation{};
        std::uint32_t has_numeric_range{};
        float numeric_min{};
        float numeric_max{};
        float numeric_step{};
        std::uint64_t unsigned_min{};
        std::uint64_t unsigned_max{};
        std::uint64_t unsigned_step{};
    };

    struct SpectraSceneControlAction {
        const char* id{};
        const char* label{};
        const char* description{};
        const char* section_id{};
        RawSpan options{};
    };

    struct SpectraSceneControlMetric {
        const char* key{};
        const char* label{};
        const char* value{};
        const char* section_id{};
        std::uint32_t display_flags{};
        std::uint32_t has_color{};
        float color[4]{};
    };

    struct SpectraSceneControlActionState {
        const char* action_id{};
        std::uint32_t enabled{};
        const char* disabled_reason{};
    };

    struct SpectraSceneControlSettingState {
        const char* key{};
        const char* value{};
        std::uint32_t has_unsigned_range{};
        std::uint64_t unsigned_min{};
        std::uint64_t unsigned_max{};
        std::uint64_t unsigned_step{};
    };

    struct SpectraSceneControlStateView {
        std::uint64_t struct_size{};
        const char* phase{};
        const char* headline{};
        const char* detail{};
        RawSpan metrics{};
        RawSpan action_states{};
        RawSpan setting_states{};
    };

    struct SpectraSceneUpdateInfo {
        std::uint64_t struct_size{};
        double wall_delta_seconds{};
        double update_delta_seconds{};
        double timeline_time_seconds{};
        std::uint64_t timeline_frame_index{};
        std::uint32_t frame_slot_index{};
        std::uint32_t frame_slot_count{};
        std::uint32_t update_running{};
    };

    struct SpectraSceneGpuDeviceIdentity {
        std::uint32_t vendor_id{};
        std::uint32_t device_id{};
        std::uint8_t device_uuid[16]{};
        std::uint8_t device_luid[8]{};
        std::uint32_t device_node_mask{};
    };

    struct SpectraSceneGpuBufferRequest {
        std::uint64_t struct_size{};
        std::uint32_t kind{};
        std::uint64_t byte_size{};
    };

    struct SpectraSceneGpuBufferSlotAllocation {
        std::uint32_t handle_kind{};
        std::uintptr_t handle{};
    };

    struct SpectraSceneGpuBufferSlotAllocationSpan {
        const SpectraSceneGpuBufferSlotAllocation* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneGpuBufferAllocation {
        std::uint64_t struct_size{};
        std::uint64_t resource_id{};
        std::uint64_t byte_size{};
        std::uint32_t kind{};
        SpectraSceneGpuBufferSlotAllocationSpan slots{};
        SpectraSceneGpuDeviceIdentity device_identity{};
    };

    typedef SpectraSceneResult (*RequestGpuBufferFn)(void*, const SpectraSceneGpuBufferRequest*, SpectraSceneGpuBufferAllocation*);
    typedef SpectraSceneResult (*ReleaseGpuBufferFn)(void*, std::uint64_t);
    typedef const char* (*HostLastErrorFn)(void*);

    struct SpectraSceneHostServices {
        std::uint64_t struct_size{};
        void* user_data{};
        RequestGpuBufferFn request_gpu_buffer{};
        ReleaseGpuBufferFn release_gpu_buffer{};
        HostLastErrorFn last_error{};
    };

    struct SpectraSceneOpenInfo {
        std::uint64_t struct_size{};
        const char* plugin_path{};
        SpectraSceneOptionSpan options{};
        const SpectraSceneHostServices* host_services{};
    };

    struct SpectraSceneTransform {
        float position[3]{};
        float rotation[4]{};
        float scale[3]{};
    };

    struct SpectraSceneCameraImage {
        const std::uint8_t* rgba8{};
        std::uint64_t rgba8_size{};
        std::uint64_t revision{};
        std::uint32_t width{};
        std::uint32_t height{};
    };

    struct SpectraSceneCamera {
        const char* name{};
        float position[3]{};
        float right[3]{};
        float down[3]{};
        float forward[3]{};
        std::uint32_t projection{};
        float vertical_fov_degrees{};
        std::uint32_t image_width{};
        std::uint32_t image_height{};
        float fx{};
        float fy{};
        float cx{};
        float cy{};
        float near_plane{};
        float far_plane{};
        std::uint32_t has_image{};
        SpectraSceneCameraImage image{};
    };

    struct SpectraSceneEntityRef {
        std::uint32_t kind{};
        const char* name{};
    };

    struct SpectraSceneViewportSegmentSet {
        const char* name{};
        SpectraSceneEntityRef owner{};
        RawSpan segments{};
        RawSpan colors{};
        RawSpan widths{};
        std::uint32_t source_kind{};
        std::uint64_t segment_count{};
        std::uint64_t buffer_id{};
        std::uint64_t source_byte_size{};
        float width{};
        std::uint32_t width_mode{};
        std::uint32_t depth_mode{};
        SpectraSceneTransform transform{};
    };

    struct SpectraSceneItems {
        RawSpan materials{};
        RawSpan lights{};
        RawSpan cameras{};
        RawSpan meshes{};
        RawSpan spheres{};
        RawSpan point_clouds{};
        RawSpan volumes{};
        RawSpan viewport_segment_sets{};
        RawSpan viewport_voxel_grids{};
    };

    struct SpectraSceneTimeline {
        std::uint32_t kind{};
        double frame_rate{};
        std::uint64_t frame_count{};
    };

    struct SpectraSceneUpdateDescriptor {
        std::uint32_t enabled{};
        std::uint32_t initial_running{};
        double step_delta_seconds{};
    };

    struct SpectraSceneViewportNavigationTarget {
        std::uint64_t revision{};
        float focus[3]{};
        float bounds_minimum[3]{};
        float bounds_maximum[3]{};
        float navigation_up[3]{};
    };

    struct SpectraSceneDocumentView {
        std::uint64_t struct_size{};
        SpectraSceneTimeline timeline{};
        SpectraSceneUpdateDescriptor update{};
        SpectraSceneViewportNavigationTarget navigation_target{};
        const char* active_camera_name{};
        SpectraSceneItems items{};
    };

    struct SpectraSceneFrameInfo {
        double time_seconds{};
        std::uint64_t frame_index{};
    };

    struct SpectraSceneFrameView {
        std::uint64_t struct_size{};
        SpectraSceneItems items{};
    };

    typedef SpectraSceneResult (*CreateFn)(const SpectraSceneOpenInfo*, SpectraSceneInstance**);
    typedef void (*DestroyFn)(SpectraSceneInstance*);
    typedef SpectraSceneResult (*UpdateFn)(SpectraSceneInstance*, const SpectraSceneUpdateInfo*);
    typedef SpectraSceneResult (*DocumentFn)(SpectraSceneInstance*, SpectraSceneDocumentView*);
    typedef SpectraSceneResult (*FrameFn)(SpectraSceneInstance*, SpectraSceneFrameInfo, SpectraSceneFrameView*);
    typedef SpectraSceneResult (*RevisionFn)(SpectraSceneInstance*, std::uint64_t*);
    typedef SpectraSceneResult (*ControlActionFn)(SpectraSceneInstance*, const char*, SpectraSceneOptionSpan);
    typedef SpectraSceneResult (*ControlSettingFn)(SpectraSceneInstance*, const char*, const char*);
    typedef SpectraSceneResult (*ControlStateFn)(SpectraSceneInstance*, SpectraSceneControlStateView*);
    typedef const char* (*LastErrorFn)(SpectraSceneInstance*);

    struct SpectraScenePlugin {
        std::uint32_t abi_version{};
        std::uint64_t struct_size{};
        const char* id{};
        const char* title{};
        const char* open_action_label{};
        RawSpan sections{};
        RawSpan open_options{};
        RawSpan control_actions{};
        RawSpan control_settings{};
        CreateFn create{};
        DestroyFn destroy{};
        UpdateFn update{};
        DocumentFn document{};
        FrameFn frame{};
        RevisionFn scene_revision{};
        ControlActionFn control_action{};
        ControlSettingFn control_setting_update{};
        ControlStateFn control_state{};
        LastErrorFn last_error{};
    };

    template <typename Project>
    struct ExportState {
        struct SceneCache {
            Document document{};
            std::vector<SpectraSceneCamera> cameras{};
            std::vector<SpectraSceneViewportSegmentSet> segments{};
        };

        struct ControlCache {
            ControlState state{};
            std::vector<SpectraSceneControlMetric> metrics{};
            std::vector<SpectraSceneControlActionState> actions{};
            std::vector<SpectraSceneControlSettingState> settings{};
        };

        struct Instance {
            Project project;
            std::string error{};
            SceneCache document{};
            SceneCache frame{};
            ControlCache controls{};
        };

        const PluginDefinition<Project>& definition;
        std::vector<SpectraSceneControlSection> sections{};
        std::vector<SpectraSceneControlOptionSchema> open_options{};
        std::vector<SpectraSceneControlAction> actions{};
        std::vector<SpectraSceneControlOptionSchema> settings{};
        SpectraScenePlugin plugin{};

        explicit ExportState(const PluginDefinition<Project>& definition) : definition(definition) {
            for (const Section& section : definition.sections) this->sections.push_back({.id = section.id.c_str(), .label = section.label.c_str()});
            for (const OptionSchema& option : definition.open_options) this->open_options.push_back(option_view(option));
            for (const ActionBinding<Project>& action : definition.actions) this->actions.push_back({.id = action.schema.id.c_str(), .label = action.schema.label.c_str(), .description = action.schema.description.c_str(), .section_id = action.schema.section_id.c_str()});
            for (const SettingBinding<Project>& setting : definition.settings) this->settings.push_back(option_view(setting.schema));
            this->plugin = SpectraScenePlugin{
                .abi_version = plugin_abi_version,
                .struct_size = sizeof(SpectraScenePlugin),
                .id = definition.id.c_str(),
                .title = definition.title.c_str(),
                .open_action_label = definition.open_action_label.c_str(),
                .sections = RawSpan{.data = this->sections.data(), .count = this->sections.size()},
                .open_options = RawSpan{.data = this->open_options.data(), .count = this->open_options.size()},
                .control_actions = RawSpan{.data = this->actions.data(), .count = this->actions.size()},
                .control_settings = RawSpan{.data = this->settings.data(), .count = this->settings.size()},
                .create = &create,
                .destroy = &destroy,
                .update = &update,
                .document = &document,
                .frame = &frame,
                .scene_revision = &revision,
                .control_action = &control_action,
                .control_setting_update = &control_setting,
                .control_state = &control_state,
                .last_error = &last_error,
            };
        }

        static SpectraSceneControlOptionSchema option_view(const OptionSchema& option) {
            return SpectraSceneControlOptionSchema{
                .key = option.key.c_str(),
                .label = option.label.c_str(),
                .description = option.description.c_str(),
                .kind = static_cast<std::uint32_t>(option.kind),
                .required = 0u,
                .default_value = option.default_value.c_str(),
                .section_id = option.section_id.c_str(),
                .presentation = option.slider ? 1u : 0u,
                .has_numeric_range = option.slider ? 1u : 0u,
                .numeric_min = option.numeric_min,
                .numeric_max = option.numeric_max,
                .numeric_step = option.numeric_step,
                .unsigned_min = option.unsigned_min,
                .unsigned_max = option.unsigned_max,
                .unsigned_step = option.unsigned_step,
            };
        }

        static SpectraSceneTransform transform() {
            SpectraSceneTransform value{};
            value.rotation[3] = 1.0F;
            value.scale[0] = 1.0F;
            value.scale[1] = 1.0F;
            value.scale[2] = 1.0F;
            return value;
        }

        static SpectraSceneCamera camera_view(const Camera& camera) {
            SpectraSceneCamera view{.name = camera.name.c_str(), .projection = 0u, .vertical_fov_degrees = camera.vertical_fov_degrees, .near_plane = camera.near_plane, .far_plane = camera.far_plane};
            std::ranges::copy(camera.position, view.position);
            std::ranges::copy(camera.right, view.right);
            std::ranges::copy(camera.down, view.down);
            std::ranges::copy(camera.forward, view.forward);
            return view;
        }

        static SpectraSceneViewportSegmentSet segment_view(const ViewportSegmentSet& segment) {
            return SpectraSceneViewportSegmentSet{
                .name = segment.name.c_str(),
                .owner = SpectraSceneEntityRef{.kind = 4u, .name = segment.owner_name.c_str()},
                .source_kind = 1u,
                .segment_count = segment.segment_count,
                .buffer_id = segment.buffer_id,
                .source_byte_size = segment.source_byte_size,
                .width = segment.width,
                .width_mode = 0u,
                .depth_mode = segment.overlay ? 1u : 0u,
                .transform = transform(),
            };
        }

        static SpectraSceneItems scene_items(SceneCache& cache) {
            cache.cameras.clear();
            cache.segments.clear();
            for (const Camera& camera : cache.document.cameras) cache.cameras.push_back(camera_view(camera));
            for (const ViewportSegmentSet& segment : cache.document.viewport_segment_sets) cache.segments.push_back(segment_view(segment));
            return SpectraSceneItems{
                .cameras = RawSpan{.data = cache.cameras.data(), .count = cache.cameras.size()},
                .viewport_segment_sets = RawSpan{.data = cache.segments.data(), .count = cache.segments.size()},
            };
        }

        static Instance& instance(SpectraSceneInstance* value) {
            return *static_cast<Instance*>(value);
        }

        static SpectraSceneResult create(const SpectraSceneOpenInfo* open_info, SpectraSceneInstance** output) noexcept {
            try {
                plugin_export_error.clear();
                std::vector<Option> options{};
                for (std::uint64_t index = 0u; index < open_info->options.count; ++index) options.push_back({.key = open_info->options.data[index].key, .value = open_info->options.data[index].value});
                auto host = std::make_shared<HostServices>();
                const SpectraSceneHostServices* raw_host = open_info->host_services;
                host->request_gpu_buffer = [raw_host](const std::uint32_t kind, const std::uint64_t byte_size) {
                    const SpectraSceneGpuBufferRequest request{.struct_size = sizeof(SpectraSceneGpuBufferRequest), .kind = kind, .byte_size = byte_size};
                    SpectraSceneGpuBufferAllocation allocation{};
                    if (raw_host->request_gpu_buffer(raw_host->user_data, &request, &allocation) != ResultOk) throw std::runtime_error(raw_host->last_error(raw_host->user_data));
                    GpuBufferAllocation decoded{.resource_id = allocation.resource_id, .byte_size = allocation.byte_size};
                    decoded.slots.reserve(allocation.slots.count);
                    for (std::uint64_t index = 0u; index < allocation.slots.count; ++index) decoded.slots.push_back({.handle_kind = static_cast<GpuResourceHandleKind>(allocation.slots.data[index].handle_kind), .handle = allocation.slots.data[index].handle});
                    return decoded;
                };
                host->release_gpu_buffer = [raw_host](const std::uint64_t resource_id) {
                    if (raw_host->release_gpu_buffer(raw_host->user_data, resource_id) != ResultOk) throw std::runtime_error(raw_host->last_error(raw_host->user_data));
                };
                Instance* created = new Instance{.project = Project::open(OpenContext{.options = std::move(options), .host_services = std::move(host)})};
                *output = static_cast<SpectraSceneInstance*>(created);
                return ResultOk;
            } catch (const std::exception& error) {
                plugin_export_error = error.what();
                return ResultError;
            }
        }

        static void destroy(SpectraSceneInstance* value) noexcept {
            delete static_cast<Instance*>(value);
        }

        static SpectraSceneResult update(SpectraSceneInstance* value, const SpectraSceneUpdateInfo* info) noexcept {
            try {
                Instance& current = instance(value);
                current.error.clear();
                current.project.update(UpdateInfo{.wall_delta_seconds = info->wall_delta_seconds, .update_delta_seconds = info->update_delta_seconds, .timeline_time_seconds = info->timeline_time_seconds, .timeline_frame_index = info->timeline_frame_index, .frame_slot_index = info->frame_slot_index, .frame_slot_count = info->frame_slot_count, .update_running = info->update_running != 0u});
                return ResultOk;
            } catch (const std::exception& error) {
                instance(value).error = error.what();
                return ResultError;
            }
        }

        static SpectraSceneResult document(SpectraSceneInstance* value, SpectraSceneDocumentView* output) noexcept {
            try {
                Instance& current = instance(value);
                SceneBuilder builder{};
                current.project.write_document(builder);
                current.document.document = builder.take_document();
                const Document& document = current.document.document;
                output->struct_size = sizeof(SpectraSceneDocumentView);
                output->timeline = SpectraSceneTimeline{};
                output->update = SpectraSceneUpdateDescriptor{.enabled = document.update.enabled ? 1u : 0u, .initial_running = document.update.initial_running ? 1u : 0u, .step_delta_seconds = document.update.step_delta_seconds};
                output->navigation_target.revision = document.navigation_target.revision;
                std::ranges::copy(document.navigation_target.focus, output->navigation_target.focus);
                std::ranges::copy(document.navigation_target.bounds_minimum, output->navigation_target.bounds_minimum);
                std::ranges::copy(document.navigation_target.bounds_maximum, output->navigation_target.bounds_maximum);
                std::ranges::copy(document.navigation_target.navigation_up, output->navigation_target.navigation_up);
                output->active_camera_name = document.active_camera_name.c_str();
                output->items = scene_items(current.document);
                return ResultOk;
            } catch (const std::exception& error) {
                instance(value).error = error.what();
                return ResultError;
            }
        }

        static SpectraSceneResult frame(SpectraSceneInstance* value, const SpectraSceneFrameInfo info, SpectraSceneFrameView* output) noexcept {
            try {
                Instance& current = instance(value);
                SceneBuilder builder{};
                current.project.write_frame(builder, FrameInfo{.time_seconds = info.time_seconds, .frame_index = info.frame_index});
                current.frame.document = builder.take_document();
                output->struct_size = sizeof(SpectraSceneFrameView);
                output->items = scene_items(current.frame);
                return ResultOk;
            } catch (const std::exception& error) {
                instance(value).error = error.what();
                return ResultError;
            }
        }

        static SpectraSceneResult revision(SpectraSceneInstance* value, std::uint64_t* output) noexcept {
            try {
                *output = instance(value).project.revision;
                return ResultOk;
            } catch (const std::exception& error) {
                instance(value).error = error.what();
                return ResultError;
            }
        }

        static SpectraSceneResult control_action(SpectraSceneInstance* value, const char* action_id, SpectraSceneOptionSpan) noexcept {
            try {
                Instance& current = instance(value);
                const std::vector<ActionBinding<Project>>& actions = Project::plugin().actions;
                const auto found = std::ranges::find_if(actions, [action_id](const ActionBinding<Project>& action) { return action.schema.id == action_id; });
                found->invoke(current.project);
                return ResultOk;
            } catch (const std::exception& error) {
                instance(value).error = error.what();
                return ResultError;
            }
        }

        static SpectraSceneResult control_setting(SpectraSceneInstance* value, const char* key, const char* setting_value) noexcept {
            try {
                Instance& current = instance(value);
                const std::vector<SettingBinding<Project>>& settings = Project::plugin().settings;
                const auto found = std::ranges::find_if(settings, [key](const SettingBinding<Project>& setting) { return setting.schema.key == key; });
                found->update(current.project, setting_value);
                return ResultOk;
            } catch (const std::exception& error) {
                instance(value).error = error.what();
                return ResultError;
            }
        }

        static SpectraSceneResult control_state(SpectraSceneInstance* value, SpectraSceneControlStateView* output) noexcept {
            try {
                Instance& current = instance(value);
                ControlBuilder builder{};
                current.project.write_controls(builder);
                current.controls.state = builder.take_state();
                current.controls.metrics.clear();
                current.controls.actions.clear();
                current.controls.settings.clear();
                for (const Metric& metric : current.controls.state.metrics) current.controls.metrics.push_back(SpectraSceneControlMetric{.key = metric.key.c_str(), .label = metric.label.c_str(), .value = metric.value.c_str(), .section_id = metric.section_id.c_str(), .display_flags = 0u, .has_color = 1u, .color = {1.0F, 1.0F, 1.0F, 1.0F}});
                for (const std::string& action_id : current.controls.state.enabled_actions) current.controls.actions.push_back({.action_id = action_id.c_str(), .enabled = 1u, .disabled_reason = ""});
                for (const SettingState& setting : current.controls.state.settings) current.controls.settings.push_back({.key = setting.key.c_str(), .value = setting.value.c_str(), .has_unsigned_range = setting.has_unsigned_range ? 1u : 0u, .unsigned_min = setting.unsigned_min, .unsigned_max = setting.unsigned_max, .unsigned_step = setting.unsigned_step});
                *output = SpectraSceneControlStateView{.struct_size = sizeof(SpectraSceneControlStateView), .phase = current.controls.state.phase.c_str(), .headline = current.controls.state.headline.c_str(), .detail = current.controls.state.detail.c_str(), .metrics = RawSpan{.data = current.controls.metrics.data(), .count = current.controls.metrics.size()}, .action_states = RawSpan{.data = current.controls.actions.data(), .count = current.controls.actions.size()}, .setting_states = RawSpan{.data = current.controls.settings.data(), .count = current.controls.settings.size()}};
                return ResultOk;
            } catch (const std::exception& error) {
                instance(value).error = error.what();
                return ResultError;
            }
        }

        static const char* last_error(SpectraSceneInstance* value) noexcept {
            if (value == nullptr) return plugin_export_error.c_str();
            return instance(value).error.c_str();
        }
    };

    export template <typename Project>
    [[nodiscard]] const SpectraScenePlugin* export_plugin() {
        static ExportState<Project> state{Project::plugin()};
        return &state.plugin;
    }

} // namespace xayah::spectra::plugin
