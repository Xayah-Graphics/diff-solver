#ifndef XAYAH_CLOTH_CUDA_TYPES_H
#define XAYAH_CLOTH_CUDA_TYPES_H

namespace xayah::cloth::cuda_kernel {
    struct Field {
        float* x;
        float* y;
        float* z;
    };

    struct ConstField {
        const float* x;
        const float* y;
        const float* z;
    };
}

#endif
