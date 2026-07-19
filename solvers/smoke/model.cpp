module;

#include "operators.h"
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

    Model::Model(Configuration next_configuration) : configuration(std::move(next_configuration)), source_{}, force_{}, velocity_{}, projection_{}, scalar_advection_{} {}

    ExecutionContext Model::allocate_context(const ExecutionMode mode) const {
        ExecutionContext context{};
        context.resource                               = std::make_shared<cuda::Resource>();
        context.cell_count_                            = static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * configuration.resolution[2];
        context.face_counts_                           = {static_cast<std::size_t>(configuration.resolution[0] + 1u) * configuration.resolution[1] * configuration.resolution[2], static_cast<std::size_t>(configuration.resolution[0]) * (configuration.resolution[1] + 1u) * configuration.resolution[2], static_cast<std::size_t>(configuration.resolution[0]) * configuration.resolution[1] * (configuration.resolution[2] + 1u)};
        cuda::Resource& resource                       = *context.resource;
        DeviceDomain& domain                           = context.domain;
        const std::size_t cell_count                   = context.cell_count_;
        const std::array<std::size_t, 3u>& face_counts = context.face_counts_;
        domain                                         = {
                                                    .cell_mask            = cuda::Buffer<std::uint32_t>(context.resource, cell_count),
                                                    .collider_velocity    = allocate_staggered_vector_field(context),
                                                    .collider_density     = allocate_scalar_field(context),
                                                    .collider_temperature = allocate_scalar_field(context),
                                                    .pressure_anchor      = 0u,
        };

        std::vector<std::uint32_t> cell_mask(cell_count);
        std::vector<Vector3> collider_velocity(cell_count);
        std::vector<float> collider_density(cell_count);
        std::vector<float> collider_temperature(cell_count);
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
                        cell_mask[index]            = 1u;
                        collider_velocity[index]    = collider.velocity;
                        collider_density[index]     = collider.density;
                        collider_temperature[index] = collider.temperature;
                    }
                }
            }
        }
        const std::array pressure_faces{configuration.pressure_boundary.x_min, configuration.pressure_boundary.x_max, configuration.pressure_boundary.y_min, configuration.pressure_boundary.y_max, configuration.pressure_boundary.z_min, configuration.pressure_boundary.z_max};
        const bool fixed_pressure = std::ranges::any_of(pressure_faces, [](const ScalarBoundaryFace& face) { return face.mode == ScalarBoundaryMode::fixed_value; });
        if (fixed_pressure)
            domain.pressure_anchor = static_cast<std::uint32_t>(cell_count);
        else
            while (cell_mask[domain.pressure_anchor] != 0u) ++domain.pressure_anchor;
        resource.copy_from_host(domain.cell_mask.data, cell_mask.data(), cell_mask.size() * sizeof(std::uint32_t));
        std::vector<float> x_face_velocity(face_counts[0]);
        std::vector<float> y_face_velocity(face_counts[1]);
        std::vector<float> z_face_velocity(face_counts[2]);
        for (std::uint32_t z = 0u; z < nz; ++z) {
            for (std::uint32_t y = 0u; y < ny; ++y) {
                for (std::uint32_t x = 0u; x <= nx; ++x) {
                    float sum{};
                    float count{};
                    if (x > 0u) {
                        const std::size_t cell = x - 1u + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) {
                            sum += collider_velocity[cell].x;
                            count += 1.0F;
                        }
                    }
                    if (x < nx) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) {
                            sum += collider_velocity[cell].x;
                            count += 1.0F;
                        }
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
                        if (cell_mask[cell] != 0u) {
                            sum += collider_velocity[cell].y;
                            count += 1.0F;
                        }
                    }
                    if (y < ny) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) {
                            sum += collider_velocity[cell].y;
                            count += 1.0F;
                        }
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
                        if (cell_mask[cell] != 0u) {
                            sum += collider_velocity[cell].z;
                            count += 1.0F;
                        }
                    }
                    if (z < nz) {
                        const std::size_t cell = x + static_cast<std::size_t>(nx) * (y + static_cast<std::size_t>(ny) * z);
                        if (cell_mask[cell] != 0u) {
                            sum += collider_velocity[cell].z;
                            count += 1.0F;
                        }
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

        context.raw_advected_velocity_ = allocate_staggered_vector_field(context);
        context.pressure_              = allocate_scalar_field(context);
        context.pressure_rhs_          = allocate_scalar_field(context);
        if (mode == ExecutionMode::differentiable) {
            context.sourced_density_tangent_       = allocate_scalar_field(context);
            context.sourced_temperature_tangent_   = allocate_scalar_field(context);
            context.force_tangent_                 = allocate_centered_vector_field(context);
            context.vorticity_tangent_scratch_     = allocate_vorticity_cache(context);
            context.vorticity_adjoint_scratch_     = allocate_vorticity_adjoint_cache(context);
            context.forced_velocity_tangent_       = allocate_staggered_vector_field(context);
            context.raw_advected_velocity_tangent_ = allocate_staggered_vector_field(context);
            context.advected_velocity_tangent_     = allocate_staggered_vector_field(context);
            context.pressure_tangent_              = allocate_scalar_field(context);
            context.pressure_rhs_tangent_          = allocate_scalar_field(context);
            context.sourced_density_adjoint_       = allocate_scalar_adjoint_field(context);
            context.sourced_temperature_adjoint_   = allocate_scalar_adjoint_field(context);
            context.force_adjoint_                 = allocate_centered_vector_adjoint_field(context);
            context.projected_velocity_adjoint_    = allocate_staggered_vector_adjoint_field(context);
            context.advected_velocity_adjoint_     = allocate_staggered_vector_adjoint_field(context);
            context.raw_advected_velocity_adjoint_ = allocate_staggered_vector_adjoint_field(context);
            context.forced_velocity_adjoint_       = allocate_staggered_vector_adjoint_field(context);
            context.pressure_adjoint_              = allocate_scalar_adjoint_field(context);
            context.pressure_rhs_adjoint_          = allocate_scalar_adjoint_field(context);
        }
        resource.synchronize();
        return context;
    }

    ScalarField Model::allocate_scalar_field(ExecutionContext& context) const {
        return {.values = cuda::Buffer<float>(context.resource, context.cell_count_)};
    }

    CenteredVectorField Model::allocate_centered_vector_field(ExecutionContext& context) const {
        return {.x = allocate_scalar_field(context), .y = allocate_scalar_field(context), .z = allocate_scalar_field(context)};
    }

    StaggeredVectorField Model::allocate_staggered_vector_field(ExecutionContext& context) const {
        return {.x = cuda::Buffer<float>(context.resource, context.face_counts_[0]), .y = cuda::Buffer<float>(context.resource, context.face_counts_[1]), .z = cuda::Buffer<float>(context.resource, context.face_counts_[2])};
    }

    VorticityCache Model::allocate_vorticity_cache(ExecutionContext& context) const {
        return {.centered_velocity = allocate_centered_vector_field(context), .vorticity = allocate_centered_vector_field(context), .magnitude = allocate_scalar_field(context), .normal = allocate_centered_vector_field(context), .normalizer = allocate_scalar_field(context)};
    }

    ScalarAdjointField Model::allocate_scalar_adjoint_field(ExecutionContext& context) const {
        return {.values = cuda::Buffer<double>(context.resource, context.cell_count_)};
    }

    CenteredVectorAdjointField Model::allocate_centered_vector_adjoint_field(ExecutionContext& context) const {
        return {.x = allocate_scalar_adjoint_field(context), .y = allocate_scalar_adjoint_field(context), .z = allocate_scalar_adjoint_field(context)};
    }

    StaggeredVectorAdjointField Model::allocate_staggered_vector_adjoint_field(ExecutionContext& context) const {
        return {.x = cuda::Buffer<double>(context.resource, context.face_counts_[0]), .y = cuda::Buffer<double>(context.resource, context.face_counts_[1]), .z = cuda::Buffer<double>(context.resource, context.face_counts_[2])};
    }

    VorticityAdjointCache Model::allocate_vorticity_adjoint_cache(ExecutionContext& context) const {
        return {.centered_velocity = allocate_centered_vector_adjoint_field(context), .vorticity = allocate_centered_vector_adjoint_field(context), .magnitude = allocate_scalar_adjoint_field(context), .normal = allocate_centered_vector_adjoint_field(context)};
    }

    State Model::allocate_state(ExecutionContext& context) const {
        State value{.density = allocate_scalar_field(context), .temperature = allocate_scalar_field(context), .velocity = allocate_staggered_vector_field(context)};
        context.resource->zero(value.density.values.data, value.density.values.size * sizeof(float));
        context.resource->zero(value.temperature.values.data, value.temperature.values.size * sizeof(float));
        context.resource->zero(value.velocity.x.data, value.velocity.x.size * sizeof(float));
        context.resource->zero(value.velocity.y.data, value.velocity.y.size * sizeof(float));
        context.resource->zero(value.velocity.z.data, value.velocity.z.size * sizeof(float));
        return value;
    }

    Control Model::allocate_control(ExecutionContext& context) const {
        Control value{.density_source = allocate_scalar_field(context), .temperature_source = allocate_scalar_field(context), .external_acceleration = allocate_centered_vector_field(context)};
        context.resource->zero(value.density_source.values.data, value.density_source.values.size * sizeof(float));
        context.resource->zero(value.temperature_source.values.data, value.temperature_source.values.size * sizeof(float));
        context.resource->zero(value.external_acceleration.x.values.data, value.external_acceleration.x.values.size * sizeof(float));
        context.resource->zero(value.external_acceleration.y.values.data, value.external_acceleration.y.values.size * sizeof(float));
        context.resource->zero(value.external_acceleration.z.values.data, value.external_acceleration.z.values.size * sizeof(float));
        return value;
    }

    Parameters Model::allocate_parameters(ExecutionContext& context) const {
        Parameters value{.ambient_temperature = cuda::Buffer<float>(context.resource, 1u), .density_buoyancy = cuda::Buffer<float>(context.resource, 1u), .temperature_buoyancy = cuda::Buffer<float>(context.resource, 1u), .vorticity_confinement = cuda::Buffer<float>(context.resource, 1u)};
        context.resource->zero(value.ambient_temperature.data, sizeof(float));
        context.resource->zero(value.density_buoyancy.data, sizeof(float));
        context.resource->zero(value.temperature_buoyancy.data, sizeof(float));
        context.resource->zero(value.vorticity_confinement.data, sizeof(float));
        return value;
    }

    StepCache Model::allocate_step_cache(ExecutionContext& context) const {
        return {.sourced_density = allocate_scalar_field(context), .sourced_temperature = allocate_scalar_field(context), .force = allocate_centered_vector_field(context), .vorticity = allocate_vorticity_cache(context), .forced_velocity = allocate_staggered_vector_field(context), .advected_velocity = allocate_staggered_vector_field(context)};
    }

    StateTangent Model::allocate_state_tangent(ExecutionContext& context) const {
        StateTangent value{.density = allocate_scalar_field(context), .temperature = allocate_scalar_field(context), .velocity = allocate_staggered_vector_field(context)};
        context.resource->zero(value.density.values.data, value.density.values.size * sizeof(float));
        context.resource->zero(value.temperature.values.data, value.temperature.values.size * sizeof(float));
        context.resource->zero(value.velocity.x.data, value.velocity.x.size * sizeof(float));
        context.resource->zero(value.velocity.y.data, value.velocity.y.size * sizeof(float));
        context.resource->zero(value.velocity.z.data, value.velocity.z.size * sizeof(float));
        return value;
    }

    ControlTangent Model::allocate_control_tangent(ExecutionContext& context) const {
        ControlTangent value{.density_source = allocate_scalar_field(context), .temperature_source = allocate_scalar_field(context), .external_acceleration = allocate_centered_vector_field(context)};
        context.resource->zero(value.density_source.values.data, value.density_source.values.size * sizeof(float));
        context.resource->zero(value.temperature_source.values.data, value.temperature_source.values.size * sizeof(float));
        context.resource->zero(value.external_acceleration.x.values.data, value.external_acceleration.x.values.size * sizeof(float));
        context.resource->zero(value.external_acceleration.y.values.data, value.external_acceleration.y.values.size * sizeof(float));
        context.resource->zero(value.external_acceleration.z.values.data, value.external_acceleration.z.values.size * sizeof(float));
        return value;
    }

    ParameterTangent Model::allocate_parameter_tangent(ExecutionContext& context) const {
        ParameterTangent value{.ambient_temperature = cuda::Buffer<float>(context.resource, 1u), .density_buoyancy = cuda::Buffer<float>(context.resource, 1u), .temperature_buoyancy = cuda::Buffer<float>(context.resource, 1u), .vorticity_confinement = cuda::Buffer<float>(context.resource, 1u)};
        context.resource->zero(value.ambient_temperature.data, sizeof(float));
        context.resource->zero(value.density_buoyancy.data, sizeof(float));
        context.resource->zero(value.temperature_buoyancy.data, sizeof(float));
        context.resource->zero(value.vorticity_confinement.data, sizeof(float));
        return value;
    }

    StateAdjoint Model::allocate_state_adjoint(ExecutionContext& context) const {
        StateAdjoint value{.density = allocate_scalar_adjoint_field(context), .temperature = allocate_scalar_adjoint_field(context), .velocity = allocate_staggered_vector_adjoint_field(context)};
        context.resource->zero(value.density.values.data, value.density.values.size * sizeof(double));
        context.resource->zero(value.temperature.values.data, value.temperature.values.size * sizeof(double));
        context.resource->zero(value.velocity.x.data, value.velocity.x.size * sizeof(double));
        context.resource->zero(value.velocity.y.data, value.velocity.y.size * sizeof(double));
        context.resource->zero(value.velocity.z.data, value.velocity.z.size * sizeof(double));
        return value;
    }

    ControlAdjoint Model::allocate_control_adjoint(ExecutionContext& context) const {
        ControlAdjoint value{.density_source = allocate_scalar_adjoint_field(context), .temperature_source = allocate_scalar_adjoint_field(context), .external_acceleration = allocate_centered_vector_adjoint_field(context)};
        context.resource->zero(value.density_source.values.data, value.density_source.values.size * sizeof(double));
        context.resource->zero(value.temperature_source.values.data, value.temperature_source.values.size * sizeof(double));
        context.resource->zero(value.external_acceleration.x.values.data, value.external_acceleration.x.values.size * sizeof(double));
        context.resource->zero(value.external_acceleration.y.values.data, value.external_acceleration.y.values.size * sizeof(double));
        context.resource->zero(value.external_acceleration.z.values.data, value.external_acceleration.z.values.size * sizeof(double));
        return value;
    }

    ParameterAdjoint Model::allocate_parameter_adjoint(ExecutionContext& context) const {
        ParameterAdjoint value{.ambient_temperature = cuda::Buffer<double>(context.resource, 1u), .density_buoyancy = cuda::Buffer<double>(context.resource, 1u), .temperature_buoyancy = cuda::Buffer<double>(context.resource, 1u), .vorticity_confinement = cuda::Buffer<double>(context.resource, 1u)};
        context.resource->zero(value.ambient_temperature.data, sizeof(double));
        context.resource->zero(value.density_buoyancy.data, sizeof(double));
        context.resource->zero(value.temperature_buoyancy.data, sizeof(double));
        context.resource->zero(value.vorticity_confinement.data, sizeof(double));
        return value;
    }

    void Model::copy_state(const State& source, State& destination, ExecutionContext& context) const {
        context.resource->copy_device(destination.density.values.data, source.density.values.data, source.density.values.size * sizeof(float));
        context.resource->copy_device(destination.temperature.values.data, source.temperature.values.data, source.temperature.values.size * sizeof(float));
        context.resource->copy_device(destination.velocity.x.data, source.velocity.x.data, source.velocity.x.size * sizeof(float));
        context.resource->copy_device(destination.velocity.y.data, source.velocity.y.data, source.velocity.y.size * sizeof(float));
        context.resource->copy_device(destination.velocity.z.data, source.velocity.z.data, source.velocity.z.size * sizeof(float));
    }

    void Model::copy_state_tangent(const StateTangent& source, StateTangent& destination, ExecutionContext& context) const {
        context.resource->copy_device(destination.density.values.data, source.density.values.data, source.density.values.size * sizeof(float));
        context.resource->copy_device(destination.temperature.values.data, source.temperature.values.data, source.temperature.values.size * sizeof(float));
        context.resource->copy_device(destination.velocity.x.data, source.velocity.x.data, source.velocity.x.size * sizeof(float));
        context.resource->copy_device(destination.velocity.y.data, source.velocity.y.data, source.velocity.y.size * sizeof(float));
        context.resource->copy_device(destination.velocity.z.data, source.velocity.z.data, source.velocity.z.size * sizeof(float));
    }

    void Model::copy_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        context.resource->copy_device(destination.density.values.data, source.density.values.data, source.density.values.size * sizeof(double));
        context.resource->copy_device(destination.temperature.values.data, source.temperature.values.data, source.temperature.values.size * sizeof(double));
        context.resource->copy_device(destination.velocity.x.data, source.velocity.x.data, source.velocity.x.size * sizeof(double));
        context.resource->copy_device(destination.velocity.y.data, source.velocity.y.data, source.velocity.y.size * sizeof(double));
        context.resource->copy_device(destination.velocity.z.data, source.velocity.z.data, source.velocity.z.size * sizeof(double));
    }

    void Model::accumulate_state_adjoint(const StateAdjoint& source, StateAdjoint& destination, ExecutionContext& context) const {
        const cudaStream_t stream = static_cast<cudaStream_t>(context.resource->native_stream);
        cuda_kernels::accumulate(stream, source.density.values.data, destination.density.values.data, source.density.values.size);
        cuda_kernels::accumulate(stream, source.temperature.values.data, destination.temperature.values.data, source.temperature.values.size);
        cuda_kernels::accumulate(stream, source.velocity.x.data, destination.velocity.x.data, source.velocity.x.size);
        cuda_kernels::accumulate(stream, source.velocity.y.data, destination.velocity.y.data, source.velocity.y.size);
        cuda_kernels::accumulate(stream, source.velocity.z.data, destination.velocity.z.data, source.velocity.z.size);
    }

    void Model::forward_step(const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& step_cache, ExecutionContext& context) const {
        source_.forward(*context.resource, configuration, context.domain, state.density, control.density_source, step_cache.sourced_density);
        source_.forward(*context.resource, configuration, context.domain, state.temperature, control.temperature_source, step_cache.sourced_temperature);
        force_.forward(*context.resource, configuration, context.domain, step_cache.sourced_density, step_cache.sourced_temperature, state.velocity, control, parameters, step_cache.force, step_cache.vorticity);
        velocity_.forward(*context.resource, configuration, context.domain, state.velocity, step_cache.force, step_cache.forced_velocity, context.raw_advected_velocity_, step_cache.advected_velocity);
        projection_.forward(*context.resource, configuration, context.domain, step_cache.advected_velocity, context.pressure_, context.pressure_rhs_, next_state.velocity);
        scalar_advection_.forward(*context.resource, configuration, context.domain, configuration.density_boundary, context.domain.collider_density, step_cache.sourced_density, next_state.velocity, next_state.density);
        scalar_advection_.forward(*context.resource, configuration, context.domain, configuration.temperature_boundary, context.domain.collider_temperature, step_cache.sourced_temperature, next_state.velocity, next_state.temperature);
    }

    void Model::jvp_step(const State& state, const Control&, const Parameters& parameters, const State& next_state, const StepCache& step_cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, ExecutionContext& context) const {
        source_.jvp(*context.resource, configuration, context.domain, state_tangent.density, control_tangent.density_source, context.sourced_density_tangent_);
        source_.jvp(*context.resource, configuration, context.domain, state_tangent.temperature, control_tangent.temperature_source, context.sourced_temperature_tangent_);
        force_.jvp(*context.resource, configuration, context.domain, step_cache.sourced_density, step_cache.sourced_temperature, state_tangent.velocity, context.sourced_density_tangent_, context.sourced_temperature_tangent_, control_tangent, parameters, parameter_tangent, context.force_tangent_, step_cache.vorticity, context.vorticity_tangent_scratch_);
        velocity_.jvp(*context.resource, configuration, context.domain, state.velocity, state_tangent.velocity, context.force_tangent_, step_cache.forced_velocity, context.forced_velocity_tangent_, context.raw_advected_velocity_tangent_, context.advected_velocity_tangent_);
        projection_.jvp(*context.resource, configuration, context.domain, context.advected_velocity_tangent_, context.pressure_tangent_, context.pressure_rhs_tangent_, next_state_tangent.velocity);
        scalar_advection_.jvp(*context.resource, configuration, context.domain, configuration.density_boundary, step_cache.sourced_density, context.sourced_density_tangent_, next_state.velocity, next_state_tangent.velocity, next_state_tangent.density);
        scalar_advection_.jvp(*context.resource, configuration, context.domain, configuration.temperature_boundary, step_cache.sourced_temperature, context.sourced_temperature_tangent_, next_state.velocity, next_state_tangent.velocity, next_state_tangent.temperature);
    }

    void Model::vjp_step(const State& state, const Control&, const Parameters& parameters, const State& next_state, const StepCache& step_cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, ExecutionContext& context) const {
        context.resource->zero(context.sourced_density_adjoint_.values.data, context.sourced_density_adjoint_.values.size * sizeof(double));
        context.resource->zero(context.sourced_temperature_adjoint_.values.data, context.sourced_temperature_adjoint_.values.size * sizeof(double));
        context.resource->zero(context.force_adjoint_.x.values.data, context.force_adjoint_.x.values.size * sizeof(double));
        context.resource->zero(context.force_adjoint_.y.values.data, context.force_adjoint_.y.values.size * sizeof(double));
        context.resource->zero(context.force_adjoint_.z.values.data, context.force_adjoint_.z.values.size * sizeof(double));
        context.resource->zero(context.projected_velocity_adjoint_.x.data, context.projected_velocity_adjoint_.x.size * sizeof(double));
        context.resource->zero(context.projected_velocity_adjoint_.y.data, context.projected_velocity_adjoint_.y.size * sizeof(double));
        context.resource->zero(context.projected_velocity_adjoint_.z.data, context.projected_velocity_adjoint_.z.size * sizeof(double));
        context.resource->zero(context.advected_velocity_adjoint_.x.data, context.advected_velocity_adjoint_.x.size * sizeof(double));
        context.resource->zero(context.advected_velocity_adjoint_.y.data, context.advected_velocity_adjoint_.y.size * sizeof(double));
        context.resource->zero(context.advected_velocity_adjoint_.z.data, context.advected_velocity_adjoint_.z.size * sizeof(double));
        context.resource->zero(context.raw_advected_velocity_adjoint_.x.data, context.raw_advected_velocity_adjoint_.x.size * sizeof(double));
        context.resource->zero(context.raw_advected_velocity_adjoint_.y.data, context.raw_advected_velocity_adjoint_.y.size * sizeof(double));
        context.resource->zero(context.raw_advected_velocity_adjoint_.z.data, context.raw_advected_velocity_adjoint_.z.size * sizeof(double));
        context.resource->zero(context.forced_velocity_adjoint_.x.data, context.forced_velocity_adjoint_.x.size * sizeof(double));
        context.resource->zero(context.forced_velocity_adjoint_.y.data, context.forced_velocity_adjoint_.y.size * sizeof(double));
        context.resource->zero(context.forced_velocity_adjoint_.z.data, context.forced_velocity_adjoint_.z.size * sizeof(double));
        context.resource->copy_device(context.projected_velocity_adjoint_.x.data, next_state_adjoint.velocity.x.data, next_state_adjoint.velocity.x.size * sizeof(double));
        context.resource->copy_device(context.projected_velocity_adjoint_.y.data, next_state_adjoint.velocity.y.data, next_state_adjoint.velocity.y.size * sizeof(double));
        context.resource->copy_device(context.projected_velocity_adjoint_.z.data, next_state_adjoint.velocity.z.data, next_state_adjoint.velocity.z.size * sizeof(double));
        scalar_advection_.vjp(*context.resource, configuration, context.domain, configuration.density_boundary, step_cache.sourced_density, next_state.velocity, next_state_adjoint.density, context.sourced_density_adjoint_, context.projected_velocity_adjoint_);
        scalar_advection_.vjp(*context.resource, configuration, context.domain, configuration.temperature_boundary, step_cache.sourced_temperature, next_state.velocity, next_state_adjoint.temperature, context.sourced_temperature_adjoint_, context.projected_velocity_adjoint_);
        projection_.vjp(*context.resource, configuration, context.domain, context.projected_velocity_adjoint_, context.pressure_adjoint_, context.pressure_rhs_adjoint_, context.advected_velocity_adjoint_);
        velocity_.vjp(*context.resource, configuration, context.domain, step_cache.forced_velocity, context.advected_velocity_adjoint_, context.raw_advected_velocity_adjoint_, context.forced_velocity_adjoint_, previous_state_adjoint.velocity, context.force_adjoint_);
        force_.vjp(*context.resource, configuration, context.domain, step_cache.sourced_density, step_cache.sourced_temperature, state.velocity, parameters, context.force_adjoint_, step_cache.vorticity, context.sourced_density_adjoint_, context.sourced_temperature_adjoint_, previous_state_adjoint.velocity, control_adjoint, parameter_adjoint, context.vorticity_adjoint_scratch_);
        source_.vjp(*context.resource, configuration, context.domain, context.sourced_density_adjoint_, previous_state_adjoint.density, control_adjoint.density_source);
        source_.vjp(*context.resource, configuration, context.domain, context.sourced_temperature_adjoint_, previous_state_adjoint.temperature, control_adjoint.temperature_source);
    }

} // namespace xayah::smoke
