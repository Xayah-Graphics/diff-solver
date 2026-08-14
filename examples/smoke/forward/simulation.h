#ifndef XAYAH_EXAMPLES_SMOKE_FORWARD_SIMULATION_H
#define XAYAH_EXAMPLES_SMOKE_FORWARD_SIMULATION_H

#include <cstdint>
#include <cuda_runtime_api.h>
#include <smoke/cuda_types.h>

namespace xayah::smoke::examples::forward::simulation_cuda {
    void launch_write_control(cudaStream_t stream, cuda_kernels::Grid grid, std::uint64_t step, float pulse_period, cuda_kernels::Vector left_center, cuda_kernels::Vector right_center, float source_radius, float density_source_rate, float temperature_source_rate, cuda_kernels::Vector left_acceleration, cuda_kernels::Vector right_acceleration, float* density_source, float* temperature_source, cuda_kernels::CenteredVectorView external_acceleration);
    void launch_reduce_metrics(cudaStream_t stream, cuda_kernels::Grid grid, const std::uint32_t* cell_mask, const float* density, const float* temperature, cuda_kernels::ConstStaggeredVectorView pre_projection_velocity, cuda_kernels::ConstStaggeredVectorView post_projection_velocity, double* metrics);

} // namespace xayah::smoke::examples::forward::simulation_cuda

#endif
