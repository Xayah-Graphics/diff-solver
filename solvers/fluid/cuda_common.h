#ifndef XAYAH_FLUID_CUDA_COMMON_H
#define XAYAH_FLUID_CUDA_COMMON_H

#include "sph.h"

#include <cuda_runtime.h>

namespace xayah::fluid {
    struct Float3 {
        float x;
        float y;
        float z;
    };

    struct Double3 {
        double x;
        double y;
        double z;
    };

    struct CellRange {
        std::uint32_t first;
        std::uint32_t last;
        std::uint32_t boundary_first;
        std::uint32_t boundary_last;
        bool valid;
    };

    __device__ inline Float3 add(const Float3 first, const Float3 second) {
        return {first.x + second.x, first.y + second.y, first.z + second.z};
    }

    __device__ inline Double3 add(const Double3 first, const Double3 second) {
        return {first.x + second.x, first.y + second.y, first.z + second.z};
    }

    __device__ inline Float3 subtract(const Float3 first, const Float3 second) {
        return {first.x - second.x, first.y - second.y, first.z - second.z};
    }

    __device__ inline Double3 subtract(const Double3 first, const Double3 second) {
        return {first.x - second.x, first.y - second.y, first.z - second.z};
    }

    __device__ inline Float3 scale(const Float3 value, const float factor) {
        return {factor * value.x, factor * value.y, factor * value.z};
    }

    __device__ inline Double3 scale(const Float3 value, const double factor) {
        return {factor * value.x, factor * value.y, factor * value.z};
    }

    __device__ inline Double3 scale(const Double3 value, const double factor) {
        return {factor * value.x, factor * value.y, factor * value.z};
    }

    __device__ inline float dot(const Float3 first, const Float3 second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    __device__ inline double dot(const Double3 first, const Float3 second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    __device__ inline double dot(const Double3 first, const Double3 second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    __device__ inline Float3 cross(const Float3 first, const Float3 second) {
        return {first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x};
    }

    __device__ inline Double3 cross(const Float3 first, const Double3 second) {
        return {first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x};
    }

    __device__ inline Double3 cross(const Double3 first, const Float3 second) {
        return {first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x};
    }

    __device__ inline float length(const Float3 value) {
        return sqrtf(dot(value, value));
    }

    __device__ inline std::uint32_t cell_index(const sph::cuda_kernel::Neighborhood neighborhood, const int x, const int y, const int z) {
        return (static_cast<std::uint32_t>(z) * neighborhood.cells_y + static_cast<std::uint32_t>(y)) * neighborhood.cells_x + static_cast<std::uint32_t>(x);
    }

    __device__ inline CellRange cell_range(const sph::cuda_kernel::Neighborhood neighborhood, const int x, const int y, const int z) {
        if (x < 0 || x >= static_cast<int>(neighborhood.cells_x) || y < 0 || y >= static_cast<int>(neighborhood.cells_y) || z < 0 || z >= static_cast<int>(neighborhood.cells_z)) return {};
        const std::uint32_t cell = cell_index(neighborhood, x, y, z);
        return {
            .first          = neighborhood.cell_offsets[cell],
            .last           = neighborhood.cell_offsets[cell + 1u],
            .boundary_first = neighborhood.boundary_cell_offsets[cell],
            .boundary_last  = neighborhood.boundary_cell_offsets[cell + 1u],
            .valid          = true,
        };
    }

    __device__ inline void particle_cell(const sph::cuda_kernel::Neighborhood neighborhood, const Float3 position, int& x, int& y, int& z) {
        x = min(static_cast<int>(neighborhood.cells_x) - 1, max(0, __float2int_rd((position.x - neighborhood.origin_x) / neighborhood.cell_size)));
        y = min(static_cast<int>(neighborhood.cells_y) - 1, max(0, __float2int_rd((position.y - neighborhood.origin_y) / neighborhood.cell_size)));
        z = min(static_cast<int>(neighborhood.cells_z) - 1, max(0, __float2int_rd((position.z - neighborhood.origin_z) / neighborhood.cell_size)));
    }
}

#endif
