module;

#include "kernels.cuh"
#include <cuda_runtime_api.h>

module xayah.smoke.model;

import std;
import xayah.cuda;
import xayah.smoke.data;
import xayah.smoke.operators;

namespace xayah::smoke {

    namespace {

        bool contains(const Ellipsoid& ellipsoid, const Vector3 point) {
            const float x = (point.x - ellipsoid.center.x) / ellipsoid.radius.x;
            const float y = (point.y - ellipsoid.center.y) / ellipsoid.radius.y;
            const float z = (point.z - ellipsoid.center.z) / ellipsoid.radius.z;
            return x * x + y * y + z * z <= 1.0F;
        }

        bool contains(const Box& box, const Vector3 point) {
            return std::abs(point.x - box.center.x) <= box.half_extent.x && std::abs(point.y - box.center.y) <= box.half_extent.y && std::abs(point.z - box.center.z) <= box.half_extent.z;
        }

    } // namespace

    ExecutionContext::ExecutionContext(const Configuration& configuration, const ExecutionMode mode) : resource_owner_(std::make_shared<cuda::Resource>()), resource(*resource_owner_), domain{}, cell_count_(static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * configuration.resolution[2]), face_counts_{static_cast<std::size_t>(configuration.resolution[0] + 1u) * configuration.resolution[1] * configuration.resolution[2], static_cast<std::size_t>(configuration.resolution[0]) * (configuration.resolution[1] + 1u) * configuration.resolution[2], static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * (configuration.resolution[2] + 1u)} {
        domain = {
            .cell_mask = make_index_buffer(cell_count_),
            .collider_velocity = make_staggered_vector_field(),
            .collider_density = make_scalar_field(),
            .collider_temperature = make_scalar_field(),
            .pressure_anchor = 0u,
        };

        std::vector<std::uint32_t> cell_mask(cell_count_);
        std::vector<Vector3> collider_velocity(cell_count_);
        std::vector<float> collider_density(cell_count_);
        std::vector<float> collider_temperature(cell_count_);
        const std::uint32_t nx = configuration.resolution[0];
        const std::uint32_t ny = configuration.resolution[1];
        const std::uint32_t nz = configuration.resolution[2];
        for (std::uint32_t z = 0u; z < nz; ++z) {
            for (std::uint32_t y = 0u; y < ny; ++y) {
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    const std::size_t index = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                    const Vector3 point{(static_cast<float>(x) + 0.5F) * configuration.cell_size, (static_cast<float>(y) + 0.5F) * configuration.cell_size, (static_cast<float>(z) + 0.5F) * configuration.cell_size};
                    for (const Collider& collider : configuration.colliders) {
                        const bool inside = std::visit([point](const auto& shape) { return contains(shape, point); }, collider.shape);
                        if (!inside) continue;
                        cell_mask[index] = 1u;
                        collider_velocity[index] = collider.velocity;
                        collider_density[index] = collider.density;
                        collider_temperature[index] = collider.temperature;
                    }
                }
            }
        }
        const std::array pressure_faces{configuration.pressure_boundary.x_min, configuration.pressure_boundary.x_max, configuration.pressure_boundary.y_min, configuration.pressure_boundary.y_max, configuration.pressure_boundary.z_min, configuration.pressure_boundary.z_max};
        const bool fixed_pressure = std::ranges::any_of(pressure_faces, [](const ScalarBoundaryFace& face) { return face.mode == ScalarBoundaryMode::fixed_value; });
        if (fixed_pressure) domain.pressure_anchor = static_cast<std::uint32_t>(cell_count_);
        else while (cell_mask[domain.pressure_anchor] != 0u) ++domain.pressure_anchor;
        resource.copy_from_host(domain.cell_mask.data, cell_mask.data(), cell_mask.size() * sizeof(std::uint32_t));
        std::vector<float> x_face_velocity(face_counts_[0]);
        std::vector<float> y_face_velocity(face_counts_[1]);
        std::vector<float> z_face_velocity(face_counts_[2]);
        for (std::uint32_t z = 0u; z < nz; ++z) {
            for (std::uint32_t y = 0u; y < ny; ++y) {
                for (std::uint32_t x = 0u; x <= nx; ++x) {
                    float sum{};
                    float count{};
                    if (x > 0u) {
                        const std::size_t cell = x - 1u + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) { sum += collider_velocity[cell].x; count += 1.0F; }
                    }
                    if (x < nx) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) { sum += collider_velocity[cell].x; count += 1.0F; }
                    }
                    x_face_velocity[x + static_cast<std::size_t>(nx + 1u) * (y + static_cast<std::size_t>(ny) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
            }
        }
        for (std::uint32_t z = 0u; z < nz; ++z) {
            for (std::uint32_t y = 0u; y <= ny; ++y) {
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    float sum{};
                    float count{};
                    if (y > 0u) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y - 1u + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) { sum += collider_velocity[cell].y; count += 1.0F; }
                    }
                    if (y < ny) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) { sum += collider_velocity[cell].y; count += 1.0F; }
                    }
                    y_face_velocity[x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny + 1u) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
            }
        }
        for (std::uint32_t z = 0u; z <= nz; ++z) {
            for (std::uint32_t y = 0u; y < ny; ++y) {
                for (std::uint32_t x = 0u; x < nx; ++x) {
                    float sum{};
                    float count{};
                    if (z > 0u) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * (z - 1u));
                        if (cell_mask[cell] != 0u) { sum += collider_velocity[cell].z; count += 1.0F; }
                    }
                    if (z < nz) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) { sum += collider_velocity[cell].z; count += 1.0F; }
                    }
                    z_face_velocity[x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z)] = count == 0.0F ? 0.0F : sum / count;
                }
            }
        }
        resource.copy_from_host(domain.collider_velocity.x.data, x_face_velocity.data(), x_face_velocity.size() * sizeof(float));
        resource.copy_from_host(domain.collider_velocity.y.data, y_face_velocity.data(), y_face_velocity.size() * sizeof(float));
        resource.copy_from_host(domain.collider_velocity.z.data, z_face_velocity.data(), z_face_velocity.size() * sizeof(float));
        resource.copy_from_host(domain.collider_density.values.data, collider_density.data(), collider_density.size() * sizeof(float));
        resource.copy_from_host(domain.collider_temperature.values.data, collider_temperature.data(), collider_temperature.size() * sizeof(float));

        raw_advected_velocity_ = make_staggered_vector_field();
        pressure_ = make_scalar_field();
        pressure_rhs_ = make_scalar_field();
        if (mode == ExecutionMode::differentiable) {
            sourced_density_tangent_ = make_scalar_field();
            sourced_temperature_tangent_ = make_scalar_field();
            force_tangent_ = make_centered_vector_field();
            vorticity_tangent_scratch_ = make_vorticity_cache();
            vorticity_adjoint_scratch_ = make_vorticity_adjoint_cache();
            forced_velocity_tangent_ = make_staggered_vector_field();
            raw_advected_velocity_tangent_ = make_staggered_vector_field();
            advected_velocity_tangent_ = make_staggered_vector_field();
            pressure_tangent_ = make_scalar_field();
            pressure_rhs_tangent_ = make_scalar_field();
            sourced_density_adjoint_ = make_scalar_adjoint_field();
            sourced_temperature_adjoint_ = make_scalar_adjoint_field();
            force_adjoint_ = make_centered_vector_adjoint_field();
            projected_velocity_adjoint_ = make_staggered_vector_adjoint_field();
            advected_velocity_adjoint_ = make_staggered_vector_adjoint_field();
            raw_advected_velocity_adjoint_ = make_staggered_vector_adjoint_field();
            forced_velocity_adjoint_ = make_staggered_vector_adjoint_field();
            pressure_adjoint_ = make_scalar_adjoint_field();
            pressure_rhs_adjoint_ = make_scalar_adjoint_field();
        }
        resource.synchronize();
    }

    void ExecutionContext::upload(const std::span<const float> source, cuda::Buffer<float>& destination) {
        if (!source.empty()) resource.copy_from_host(destination.data, source.data(), source.size_bytes());
        resource.synchronize();
    }

    void ExecutionContext::download(const cuda::Buffer<float>& source, const std::span<float> destination) {
        if (!destination.empty()) resource.copy_to_host(destination.data(), source.data, destination.size_bytes());
        resource.synchronize();
    }

    void ExecutionContext::upload(const std::span<const double> source, cuda::Buffer<double>& destination) {
        if (!source.empty()) resource.copy_from_host(destination.data, source.data(), source.size_bytes());
        resource.synchronize();
    }

    void ExecutionContext::download(const cuda::Buffer<double>& source, const std::span<double> destination) {
        if (!destination.empty()) resource.copy_to_host(destination.data(), source.data, destination.size_bytes());
        resource.synchronize();
    }

    void ExecutionContext::upload(const float source, cuda::Buffer<float>& destination) {
        resource.copy_from_host(destination.data, &source, sizeof(float));
        resource.synchronize();
    }

    void ExecutionContext::synchronize() {
        resource.synchronize();
    }

    cuda::Buffer<float> ExecutionContext::make_buffer(const std::size_t size) const {
        return cuda::Buffer<float>(resource_owner_, size);
    }

    cuda::Buffer<double> ExecutionContext::make_adjoint_buffer(const std::size_t size) const {
        return cuda::Buffer<double>(resource_owner_, size);
    }

    cuda::Buffer<std::uint32_t> ExecutionContext::make_index_buffer(const std::size_t size) const {
        return cuda::Buffer<std::uint32_t>(resource_owner_, size);
    }

    ScalarField ExecutionContext::make_scalar_field() const {
        return {.values = make_buffer(cell_count_)};
    }

    CenteredVectorField ExecutionContext::make_centered_vector_field() const {
        return {.x = make_scalar_field(), .y = make_scalar_field(), .z = make_scalar_field()};
    }

    StaggeredVectorField ExecutionContext::make_staggered_vector_field() const {
        return {.x = make_buffer(face_counts_[0]), .y = make_buffer(face_counts_[1]), .z = make_buffer(face_counts_[2])};
    }

    VorticityCache ExecutionContext::make_vorticity_cache() const {
        return {.centered_velocity = make_centered_vector_field(), .vorticity = make_centered_vector_field(), .magnitude = make_scalar_field(), .normal = make_centered_vector_field(), .normalizer = make_scalar_field()};
    }

    ScalarAdjointField ExecutionContext::make_scalar_adjoint_field() const {
        return {.values = make_adjoint_buffer(cell_count_)};
    }

    CenteredVectorAdjointField ExecutionContext::make_centered_vector_adjoint_field() const {
        return {.x = make_scalar_adjoint_field(), .y = make_scalar_adjoint_field(), .z = make_scalar_adjoint_field()};
    }

    StaggeredVectorAdjointField ExecutionContext::make_staggered_vector_adjoint_field() const {
        return {.x = make_adjoint_buffer(face_counts_[0]), .y = make_adjoint_buffer(face_counts_[1]), .z = make_adjoint_buffer(face_counts_[2])};
    }

    VorticityAdjointCache ExecutionContext::make_vorticity_adjoint_cache() const {
        return {.centered_velocity = make_centered_vector_adjoint_field(), .vorticity = make_centered_vector_adjoint_field(), .magnitude = make_scalar_adjoint_field(), .normal = make_centered_vector_adjoint_field()};
    }

    void ExecutionContext::zero(cuda::Buffer<float>& buffer) {
        if (buffer.size != 0u) resource.zero(buffer.data, buffer.size * sizeof(float));
    }

    void ExecutionContext::zero(cuda::Buffer<double>& buffer) {
        if (buffer.size != 0u) resource.zero(buffer.data, buffer.size * sizeof(double));
    }

    void ExecutionContext::zero(ScalarField& field) {
        zero(field.values);
    }

    void ExecutionContext::zero(CenteredVectorField& field) {
        zero(field.x); zero(field.y); zero(field.z);
    }

    void ExecutionContext::zero(StaggeredVectorField& field) {
        zero(field.x); zero(field.y); zero(field.z);
    }

    void ExecutionContext::zero(VorticityCache& cache) {
        zero(cache.centered_velocity); zero(cache.vorticity); zero(cache.magnitude); zero(cache.normal); zero(cache.normalizer);
    }

    void ExecutionContext::zero(ScalarAdjointField& field) {
        zero(field.values);
    }

    void ExecutionContext::zero(CenteredVectorAdjointField& field) {
        zero(field.x); zero(field.y); zero(field.z);
    }

    void ExecutionContext::zero(StaggeredVectorAdjointField& field) {
        zero(field.x); zero(field.y); zero(field.z);
    }

    void ExecutionContext::zero(VorticityAdjointCache& cache) {
        zero(cache.centered_velocity); zero(cache.vorticity); zero(cache.magnitude); zero(cache.normal);
    }

    void ExecutionContext::copy(const cuda::Buffer<float>& source, cuda::Buffer<float>& destination) {
        if (source.size != 0u) resource.copy_device(destination.data, source.data, source.size * sizeof(float));
    }

    void ExecutionContext::copy(const cuda::Buffer<double>& source, cuda::Buffer<double>& destination) {
        if (source.size != 0u) resource.copy_device(destination.data, source.data, source.size * sizeof(double));
    }

    void ExecutionContext::copy(const ScalarField& source, ScalarField& destination) {
        copy(source.values, destination.values);
    }

    void ExecutionContext::copy(const StaggeredVectorField& source, StaggeredVectorField& destination) {
        copy(source.x, destination.x); copy(source.y, destination.y); copy(source.z, destination.z);
    }

    void ExecutionContext::copy(const ScalarAdjointField& source, ScalarAdjointField& destination) {
        copy(source.values, destination.values);
    }

    void ExecutionContext::copy(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) {
        copy(source.x, destination.x); copy(source.y, destination.y); copy(source.z, destination.z);
    }

    void ExecutionContext::accumulate(const ScalarAdjointField& source, ScalarAdjointField& destination) {
        cuda_kernels::accumulate(static_cast<cudaStream_t>(resource.native_stream), source.values.data, destination.values.data, source.values.size);
    }

    void ExecutionContext::accumulate(const StaggeredVectorAdjointField& source, StaggeredVectorAdjointField& destination) {
        cuda_kernels::accumulate(static_cast<cudaStream_t>(resource.native_stream), source.x.data, destination.x.data, source.x.size);
        cuda_kernels::accumulate(static_cast<cudaStream_t>(resource.native_stream), source.y.data, destination.y.data, source.y.size);
        cuda_kernels::accumulate(static_cast<cudaStream_t>(resource.native_stream), source.z.data, destination.z.data, source.z.size);
    }

    Model::Model(Configuration next_configuration) : configuration(std::move(next_configuration)), source_{}, force_{}, velocity_{}, projection_{}, scalar_advection_{} {}

    ExecutionContext Model::make_context(const ExecutionMode mode) const {
        return ExecutionContext(configuration, mode);
    }

    State Model::make_state(ExecutionContext& context) const {
        State value{.density = context.make_scalar_field(), .temperature = context.make_scalar_field(), .velocity = context.make_staggered_vector_field()};
        context.zero(value.density); context.zero(value.temperature); context.zero(value.velocity);
        return value;
    }

    Control Model::make_control(ExecutionContext& context) const {
        Control value{.density_source = context.make_scalar_field(), .temperature_source = context.make_scalar_field(), .external_acceleration = context.make_centered_vector_field()};
        context.zero(value.density_source); context.zero(value.temperature_source); context.zero(value.external_acceleration);
        return value;
    }

    Parameters Model::make_parameters(ExecutionContext& context) const {
        Parameters value{.ambient_temperature = context.make_buffer(1u), .density_buoyancy = context.make_buffer(1u), .temperature_buoyancy = context.make_buffer(1u), .vorticity_confinement = context.make_buffer(1u)};
        context.zero(value.ambient_temperature); context.zero(value.density_buoyancy); context.zero(value.temperature_buoyancy); context.zero(value.vorticity_confinement);
        return value;
    }

    StepCache Model::make_step_cache(ExecutionContext& context) const {
        return {.sourced_density = context.make_scalar_field(), .sourced_temperature = context.make_scalar_field(), .force = context.make_centered_vector_field(), .vorticity = context.make_vorticity_cache(), .forced_velocity = context.make_staggered_vector_field(), .advected_velocity = context.make_staggered_vector_field()};
    }

    VorticityAdjointCache Model::make_vorticity_adjoint_cache(ExecutionContext& context) const {
        VorticityAdjointCache value = context.make_vorticity_adjoint_cache();
        context.zero(value);
        return value;
    }

    StateTangent Model::make_state_tangent(ExecutionContext& context) const {
        StateTangent value{.density = context.make_scalar_field(), .temperature = context.make_scalar_field(), .velocity = context.make_staggered_vector_field()};
        context.zero(value.density); context.zero(value.temperature); context.zero(value.velocity);
        return value;
    }

    ControlTangent Model::make_control_tangent(ExecutionContext& context) const {
        ControlTangent value{.density_source = context.make_scalar_field(), .temperature_source = context.make_scalar_field(), .external_acceleration = context.make_centered_vector_field()};
        context.zero(value.density_source); context.zero(value.temperature_source); context.zero(value.external_acceleration);
        return value;
    }

    ParameterTangent Model::make_parameter_tangent(ExecutionContext& context) const {
        ParameterTangent value{.ambient_temperature = context.make_buffer(1u), .density_buoyancy = context.make_buffer(1u), .temperature_buoyancy = context.make_buffer(1u), .vorticity_confinement = context.make_buffer(1u)};
        context.zero(value.ambient_temperature); context.zero(value.density_buoyancy); context.zero(value.temperature_buoyancy); context.zero(value.vorticity_confinement);
        return value;
    }

    StateAdjoint Model::make_state_adjoint(ExecutionContext& context) const {
        StateAdjoint value{.density = context.make_scalar_adjoint_field(), .temperature = context.make_scalar_adjoint_field(), .velocity = context.make_staggered_vector_adjoint_field()};
        context.zero(value.density); context.zero(value.temperature); context.zero(value.velocity);
        return value;
    }

    ControlAdjoint Model::make_control_adjoint(ExecutionContext& context) const {
        ControlAdjoint value{.density_source = context.make_scalar_adjoint_field(), .temperature_source = context.make_scalar_adjoint_field(), .external_acceleration = context.make_centered_vector_adjoint_field()};
        context.zero(value.density_source); context.zero(value.temperature_source); context.zero(value.external_acceleration);
        return value;
    }

    ParameterAdjoint Model::make_parameter_adjoint(ExecutionContext& context) const {
        ParameterAdjoint value{.ambient_temperature = context.make_adjoint_buffer(1u), .density_buoyancy = context.make_adjoint_buffer(1u), .temperature_buoyancy = context.make_adjoint_buffer(1u), .vorticity_confinement = context.make_adjoint_buffer(1u)};
        context.zero(value.ambient_temperature); context.zero(value.density_buoyancy); context.zero(value.temperature_buoyancy); context.zero(value.vorticity_confinement);
        return value;
    }

    void Model::copy_state(const State& source, State& destination, ExecutionContext& context) const {
        context.copy(source.density, destination.density); context.copy(source.temperature, destination.temperature); context.copy(source.velocity, destination.velocity);
    }

    void Model::copy_state_tangent(const StateTangent& source, StateTangent& destination, ExecutionContext& context) const {
        context.copy(source.density, destination.density); context.copy(source.temperature, destination.temperature); context.copy(source.velocity, destination.velocity);
    }

    void Model::copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        context.copy(source.density, destination.density); context.copy(source.temperature, destination.temperature); context.copy(source.velocity, destination.velocity);
    }

    void Model::accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        context.accumulate(source.density, destination.density); context.accumulate(source.temperature, destination.temperature); context.accumulate(source.velocity, destination.velocity);
    }

    void Model::forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& step_cache, ExecutionContext& context) const {
        source_.forward(context.resource, configuration, context.domain, state.density, control.density_source, step_cache.sourced_density);
        source_.forward(context.resource, configuration, context.domain, state.temperature, control.temperature_source, step_cache.sourced_temperature);
        force_.forward(context.resource, configuration, context.domain, step_cache.sourced_density, step_cache.sourced_temperature, state.velocity, control, parameters, step_cache.force, step_cache.vorticity);
        velocity_.forward(context.resource, configuration, context.domain, state.velocity, step_cache.force, step_cache.forced_velocity, context.raw_advected_velocity_, step_cache.advected_velocity);
        projection_.forward(context.resource, configuration, context.domain, step_cache.advected_velocity, context.pressure_, context.pressure_rhs_, next_state.velocity);
        scalar_advection_.forward(context.resource, configuration, context.domain, configuration.density_boundary, context.domain.collider_density, step_cache.sourced_density, next_state.velocity, next_state.density);
        scalar_advection_.forward(context.resource, configuration, context.domain, configuration.temperature_boundary, context.domain.collider_temperature, step_cache.sourced_temperature, next_state.velocity, next_state.temperature);
    }

    void Model::jvp_step(const State& state, const Control&, const Parameters& parameters, const State& next_state, const StepCache& step_cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const {
        source_.jvp(context.resource, configuration, context.domain, state_tangent.density, control_tangent.density_source, context.sourced_density_tangent_);
        source_.jvp(context.resource, configuration, context.domain, state_tangent.temperature, control_tangent.temperature_source, context.sourced_temperature_tangent_);
        force_.jvp(context.resource, configuration, context.domain, step_cache.sourced_density, step_cache.sourced_temperature, state_tangent.velocity, context.sourced_density_tangent_, context.sourced_temperature_tangent_, control_tangent, parameters, parameter_tangent, context.force_tangent_, step_cache.vorticity, context.vorticity_tangent_scratch_);
        velocity_.jvp(context.resource, configuration, context.domain, state.velocity, state_tangent.velocity, context.force_tangent_, step_cache.forced_velocity, context.forced_velocity_tangent_, context.raw_advected_velocity_tangent_, context.advected_velocity_tangent_);
        projection_.jvp(context.resource, configuration, context.domain, context.advected_velocity_tangent_, context.pressure_tangent_, context.pressure_rhs_tangent_, next_state_tangent.velocity);
        scalar_advection_.jvp(context.resource, configuration, context.domain, configuration.density_boundary, step_cache.sourced_density, context.sourced_density_tangent_, next_state.velocity, next_state_tangent.velocity, next_state_tangent.density);
        scalar_advection_.jvp(context.resource, configuration, context.domain, configuration.temperature_boundary, step_cache.sourced_temperature, context.sourced_temperature_tangent_, next_state.velocity, next_state_tangent.velocity, next_state_tangent.temperature);
    }

    void Model::vjp_step(const State& state, const Control&, const Parameters& parameters, const State& next_state, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const {
        context.zero(context.sourced_density_adjoint_); context.zero(context.sourced_temperature_adjoint_); context.zero(context.force_adjoint_);
        context.zero(context.projected_velocity_adjoint_); context.zero(context.advected_velocity_adjoint_); context.zero(context.raw_advected_velocity_adjoint_); context.zero(context.forced_velocity_adjoint_);
        context.copy(next_state_adjoint.velocity, context.projected_velocity_adjoint_);
        scalar_advection_.vjp(context.resource, configuration, context.domain, configuration.density_boundary, step_cache.sourced_density, next_state.velocity, next_state_adjoint.density, context.sourced_density_adjoint_, context.projected_velocity_adjoint_);
        scalar_advection_.vjp(context.resource, configuration, context.domain, configuration.temperature_boundary, step_cache.sourced_temperature, next_state.velocity, next_state_adjoint.temperature, context.sourced_temperature_adjoint_, context.projected_velocity_adjoint_);
        projection_.vjp(context.resource, configuration, context.domain, context.projected_velocity_adjoint_, context.pressure_adjoint_, context.pressure_rhs_adjoint_, context.advected_velocity_adjoint_);
        velocity_.vjp(context.resource, configuration, context.domain, step_cache.forced_velocity, context.advected_velocity_adjoint_, context.raw_advected_velocity_adjoint_, context.forced_velocity_adjoint_, previous_state_adjoint.velocity, context.force_adjoint_);
        force_.vjp(context.resource, configuration, context.domain, step_cache.sourced_density, step_cache.sourced_temperature, state.velocity, parameters, context.force_adjoint_, step_cache.vorticity, context.sourced_density_adjoint_, context.sourced_temperature_adjoint_, previous_state_adjoint.velocity, control_adjoint, parameter_adjoint, context.vorticity_adjoint_scratch_);
        source_.vjp(context.resource, configuration, context.domain, context.sourced_density_adjoint_, previous_state_adjoint.density, control_adjoint.density_source);
        source_.vjp(context.resource, configuration, context.domain, context.sourced_temperature_adjoint_, previous_state_adjoint.temperature, control_adjoint.temperature_source);
    }

} // namespace xayah::smoke
