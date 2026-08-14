#ifndef XAYAH_SMOKE_CUDA_TYPES_H
#define XAYAH_SMOKE_CUDA_TYPES_H

#include <cstdint>

namespace xayah::smoke::cuda_kernels {
    struct Vector {
        float x;
        float y;
        float z;
    };

    struct Grid {
        std::uint32_t nx;
        std::uint32_t ny;
        std::uint32_t nz;
        float cell_size;
        float time_step;
    };

    struct ScalarBoundaryData {
        std::uint32_t modes[6];
        float values[6];
    };

    struct VelocityBoundaryData {
        std::uint32_t modes[6];
        float values[18];
    };

    struct ScalarView {
        float* values;
    };

    struct ConstScalarView {
        const float* values;
    };

    struct ConstCenteredVectorView;
    struct ConstStaggeredVectorView;
    struct ConstScalarAdjointView;
    struct ConstCenteredVectorAdjointView;
    struct ConstStaggeredVectorAdjointView;

    struct CenteredVectorView {
        float* x;
        float* y;
        float* z;

        operator ConstCenteredVectorView() const;
    };

    struct ConstCenteredVectorView {
        const float* x;
        const float* y;
        const float* z;
    };

    inline CenteredVectorView::operator ConstCenteredVectorView() const {
        return {x, y, z};
    }

    struct StaggeredVectorView {
        float* x;
        float* y;
        float* z;

        operator ConstStaggeredVectorView() const;
    };

    struct ConstStaggeredVectorView {
        const float* x;
        const float* y;
        const float* z;
    };

    inline StaggeredVectorView::operator ConstStaggeredVectorView() const {
        return {x, y, z};
    }

    struct ScalarAdjointView {
        double* values;

        operator ConstScalarAdjointView() const;
    };

    struct ConstScalarAdjointView {
        const double* values;
    };

    inline ScalarAdjointView::operator ConstScalarAdjointView() const {
        return {values};
    }

    struct CenteredVectorAdjointView {
        double* x;
        double* y;
        double* z;

        operator ConstCenteredVectorAdjointView() const;
    };

    struct ConstCenteredVectorAdjointView {
        const double* x;
        const double* y;
        const double* z;
    };

    inline CenteredVectorAdjointView::operator ConstCenteredVectorAdjointView() const {
        return {x, y, z};
    }

    struct StaggeredVectorAdjointView {
        double* x;
        double* y;
        double* z;

        operator ConstStaggeredVectorAdjointView() const;
    };

    struct ConstStaggeredVectorAdjointView {
        const double* x;
        const double* y;
        const double* z;
    };

    inline StaggeredVectorAdjointView::operator ConstStaggeredVectorAdjointView() const {
        return {x, y, z};
    }
}

#endif
