#ifndef EPICAP_GPU_KERNELS_HPP
#define EPICAP_GPU_KERNELS_HPP

#include <cuda_runtime.h>

#include "../../ep_types_gpu.h"

namespace epicap {

void uploadGammaLUT(const float* lut);

void launch_calc_kernel(const uchar3* input,
                        epPixel* output,
                        int width,
                        int height,
                        cudaStream_t stream);

void launch_resize_bilinear(const uchar3* src,
                            int srcWidth,
                            int srcHeight,
                            size_t srcStrideBytes,
                            uchar3* dst,
                            int dstWidth,
                            int dstHeight,
                            size_t dstStrideBytes,
                            cudaStream_t stream);

void launch_compare_kernel(const epPixel* prev,
                           epPixel* curr,
                           int width,
                           int height,
                           bool hdr,
                           int* harmfulLum,
                           int* harmfulCol,
                           cudaStream_t stream);

}  // namespace epicap

#endif  // EPICAP_GPU_KERNELS_HPP
