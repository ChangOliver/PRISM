#include "gpu_kernels.hpp"

#include <cuda_runtime.h>

namespace {

constexpr int LUT_SIZE = 256;

__constant__ float cGammaLUT[LUT_SIZE];
__constant__ float cXCoeffs[3] = {0.1804375f, 0.3575761f, 0.4124564f};
__constant__ float cYCoeffs[3] = {0.0721750f, 0.7151522f, 0.2126729f};
__constant__ float cZCoeffs[3] = {0.9503041f, 0.1191920f, 0.0193339f};

__device__ __forceinline__ float inverseGammaFast(float val) {
    if (val < 0.0f) return cGammaLUT[0];
    if (val >= 256.0f) return cGammaLUT[255];
    return cGammaLUT[static_cast<int>(val)];
}

__device__ __forceinline__ void load_bgr_linearRGB(const uchar3& bgr,
                                                   float& r, float& g, float& b) {
    b = inverseGammaFast(static_cast<float>(bgr.x));
    g = inverseGammaFast(static_cast<float>(bgr.y));
    r = inverseGammaFast(static_cast<float>(bgr.z));
}

__device__ __forceinline__ int warpReduceSum(int val, unsigned mask) {
#if __CUDA_ARCH__ >= 300
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
#endif
    return val;
}

__global__ void ep_calc_kernel(const uchar3* __restrict__ INPUT,
                               epPixel* __restrict__ RET,
                               int width, int height) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= height || j >= width) {
        return;
    }

    const int idx = i * width + j;
    const uchar3 pix_bgr = INPUT[idx];

    float r, g, b;
    load_bgr_linearRGB(pix_bgr, r, g, b);

    const float X = b * cXCoeffs[0] +
                    g * cXCoeffs[1] +
                    r * cXCoeffs[2];
    const float Y = b * cYCoeffs[0] +
                    g * cYCoeffs[1] +
                    r * cYCoeffs[2];
    const float Z = b * cZCoeffs[0] +
                    g * cZCoeffs[1] +
                    r * cZCoeffs[2];

    const float denom = X + 15.0f * Y + 3.0f * Z;
    const float chroma_x = (denom == 0.0f) ? 0.0f : (4.0f * X) / denom;
    const float chroma_y = (denom == 0.0f) ? 0.0f : (9.0f * Y) / denom;

    const float sum_rgb = r + g + b;
    const float red_ratio = (sum_rgb == 0.0f) ? 0.0f : (r / sum_rgb);

    RET[idx].luminance = Y;
    RET[idx].red_ratio = red_ratio;
    RET[idx].chroma_x = chroma_x;
    RET[idx].chroma_y = chroma_y;
}

__global__ void ep_compare_kernel(const epPixel* __restrict__ prev,
                                  epPixel* __restrict__ curr,
                                  int width, int height,
                                  bool HDR,
                                  int* __restrict__ harmfulLumCount,
                                  int* __restrict__ harmfulColCount) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    bool inBounds = (i < height) && (j < width);

    int localLum = 0;
    int localCol = 0;

    if (inBounds) {
        const int idx = i * width + j;
        const epPixel prevPixel = prev[idx];
        epPixel currPixel = curr[idx];
        currPixel.isIncLum = false;
        currPixel.isDecLum = false;
        currPixel.isIncCol = false;
        currPixel.isDecCol = false;

        const float I1 = prevPixel.luminance;
        const float I2 = currPixel.luminance;
        const float lumDiff = fabsf(I1 - I2);

        bool isHarmfulLum = (lumDiff > 0.1f) &&
                            (I1 < 0.8f || I2 < 0.8f);
        if (HDR && I1 > 0.8f && I2 > 0.8f) {
            const float denomHDR = I1 + I2;
            if (denomHDR > 0.0f) {
                const float ratio = lumDiff / denomHDR;
                if (ratio > 0.05882352941f) {
                    isHarmfulLum = true;
                }
            }
        }

        if (isHarmfulLum) {
            currPixel.isIncLum = (I1 < I2);
            currPixel.isDecLum = (I1 > I2);
            if ((prevPixel.isIncLum && currPixel.isDecLum) ||
                (prevPixel.isDecLum && currPixel.isIncLum)) {
                localLum = 1;
                currPixel.isIncLum = false;
                currPixel.isDecLum = false;
            }
        } else {
            currPixel.isIncLum = prevPixel.isIncLum;
            currPixel.isDecLum = prevPixel.isDecLum;
        }

        const float Rr1 = prevPixel.red_ratio;
        const float Rr2 = currPixel.red_ratio;
        bool isHarmfulCol = false;
        if (Rr1 >= 0.8f || Rr2 >= 0.8f) {
            const float dx = prevPixel.chroma_x - currPixel.chroma_x;
            const float dy = prevPixel.chroma_y - currPixel.chroma_y;
            const float dx_sq = __fmul_rn(dx, dx);
            const float dy_sq = __fmul_rn(dy, dy);
            const float colorDiff = __fadd_rn(dx_sq, dy_sq);
            if (colorDiff > 0.04f) {
                isHarmfulCol = true;
            }
        }

        if (isHarmfulCol) {
            currPixel.isIncCol = (Rr1 < Rr2);
            currPixel.isDecCol = (Rr1 > Rr2);
            if ((prevPixel.isIncCol && currPixel.isDecCol) ||
                (prevPixel.isDecCol && currPixel.isIncCol)) {
                localCol = 1;
                currPixel.isIncCol = false;
                currPixel.isDecCol = false;
            }
        } else {
            currPixel.isIncCol = prevPixel.isIncCol;
            currPixel.isDecCol = prevPixel.isDecCol;
        }

        curr[idx] = currPixel;
    }

    const unsigned activeMask = __ballot_sync(0xffffffff, inBounds);
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid & (warpSize - 1);
    int warpLum = warpReduceSum(localLum, 0xffffffff);
    int warpCol = warpReduceSum(localCol, 0xffffffff);
    if (lane == 0 && activeMask != 0) {
        if (warpLum) atomicAdd(harmfulLumCount, warpLum);
        if (warpCol) atomicAdd(harmfulColCount, warpCol);
    }
}

}  // namespace

namespace epicap {

void uploadGammaLUT(const float* lut) {
    cudaMemcpyToSymbol(cGammaLUT, lut, LUT_SIZE * sizeof(float));
}

void launch_calc_kernel(const uchar3* input,
                        epPixel* output,
                        int width,
                        int height,
                        cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    ep_calc_kernel<<<grid, block, 0, stream>>>(input, output, width, height);
}

__device__ __forceinline__ int clampToByte(float v) {
    v = fmaxf(0.0f, fminf(255.0f, v));
    return static_cast<int>(v + 0.5f);
}

__global__ void resize_bilinear_kernel(const uchar3* __restrict__ src,
                                       int srcWidth,
                                       int srcHeight,
                                       int srcPitchElems,
                                       uchar3* __restrict__ dst,
                                       int dstWidth,
                                       int dstHeight,
                                       int dstPitchElems) {
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dstWidth || dy >= dstHeight) {
        return;
    }

    const float scaleX = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    const float scaleY = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);
    float srcX = (static_cast<float>(dx) + 0.5f) * scaleX - 0.5f;
    float srcY = (static_cast<float>(dy) + 0.5f) * scaleY - 0.5f;

    srcX = fmaxf(0.0f, fminf(srcX, static_cast<float>(srcWidth - 1)));
    srcY = fmaxf(0.0f, fminf(srcY, static_cast<float>(srcHeight - 1)));

    int x0 = static_cast<int>(floorf(srcX));
    int y0 = static_cast<int>(floorf(srcY));
    int x1 = min(x0 + 1, srcWidth - 1);
    int y1 = min(y0 + 1, srcHeight - 1);

    const float fx = srcX - x0;
    const float fy = srcY - y0;

    const uchar3 p00 = src[y0 * srcPitchElems + x0];
    const uchar3 p01 = src[y0 * srcPitchElems + x1];
    const uchar3 p10 = src[y1 * srcPitchElems + x0];
    const uchar3 p11 = src[y1 * srcPitchElems + x1];

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const float b = w00 * static_cast<float>(p00.x) +
                    w01 * static_cast<float>(p01.x) +
                    w10 * static_cast<float>(p10.x) +
                    w11 * static_cast<float>(p11.x);
    const float g = w00 * static_cast<float>(p00.y) +
                    w01 * static_cast<float>(p01.y) +
                    w10 * static_cast<float>(p10.y) +
                    w11 * static_cast<float>(p11.y);
    const float r = w00 * static_cast<float>(p00.z) +
                    w01 * static_cast<float>(p01.z) +
                    w10 * static_cast<float>(p10.z) +
                    w11 * static_cast<float>(p11.z);

    const int dstIdx = dy * dstPitchElems + dx;
    dst[dstIdx] = make_uchar3(
        static_cast<unsigned char>(clampToByte(b)),
        static_cast<unsigned char>(clampToByte(g)),
        static_cast<unsigned char>(clampToByte(r)));
}

void launch_resize_bilinear(const uchar3* src,
                            int srcWidth,
                            int srcHeight,
                            size_t srcStrideBytes,
                            uchar3* dst,
                            int dstWidth,
                            int dstHeight,
                            size_t dstStrideBytes,
                            cudaStream_t stream) {
    if (!src || !dst) {
        return;
    }
    const int srcPitchElems = static_cast<int>(srcStrideBytes / sizeof(uchar3));
    const int dstPitchElems = static_cast<int>(dstStrideBytes / sizeof(uchar3));
    dim3 block(16, 16);
    dim3 grid((dstWidth + block.x - 1) / block.x,
              (dstHeight + block.y - 1) / block.y);
    resize_bilinear_kernel<<<grid, block, 0, stream>>>(
        src, srcWidth, srcHeight, srcPitchElems,
        dst, dstWidth, dstHeight, dstPitchElems);
}

void launch_compare_kernel(const epPixel* prev,
                           epPixel* curr,
                           int width,
                           int height,
                           bool hdr,
                           int* harmfulLum,
                           int* harmfulCol,
                           cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    ep_compare_kernel<<<grid, block, 0, stream>>>(
        prev, curr, width, height, hdr, harmfulLum, harmfulCol);
}

}  // namespace epicap
