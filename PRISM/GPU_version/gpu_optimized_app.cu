#include "gpu_optimized_app.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../ep_types_gpu.h"

namespace {

namespace fs = std::filesystem;

constexpr int LUT_SIZE = 256;
constexpr int HOST_BUFFER_COUNT = 3;

__constant__ float cGammaLUT[LUT_SIZE];
__constant__ float cXCoeffs[3] = {0.1804375f, 0.3575761f, 0.4124564f};
__constant__ float cYCoeffs[3] = {0.0721750f, 0.7151522f, 0.2126729f};
__constant__ float cZCoeffs[3] = {0.9503041f, 0.1191920f, 0.0193339f};

inline double computePixelDensity(int width, int height, int screenSize) {
    if (screenSize <= 0) {
        return 0.0;
    }
    return std::sqrt(static_cast<double>(width) * width +
                     static_cast<double>(height) * height) / screenSize;
}

inline double computeMinSafeArea(int resolution_h, int resolution_w,
                                 int viewingDistance, double pixelDensity) {
    if (viewingDistance <= 0) {
        return 0.25 * static_cast<double>(resolution_h) * resolution_w;
    }
    const double factor = 0.1745 * 0.1309 * 0.25;
    return viewingDistance * viewingDistance * pixelDensity * pixelDensity * factor;
}

bool loadGammaLUT(const std::string& path, std::vector<float>& lut) {
    std::vector<fs::path> candidates;
    if (!path.empty()) {
        candidates.emplace_back(path);
    } else {
        const fs::path defaultName{"inverseGammaLUT.bin"};
        candidates.push_back(defaultName);
        candidates.push_back(fs::path("../") / defaultName);
        candidates.push_back(fs::path("../../") / defaultName);
        candidates.push_back(fs::path("../../../") / defaultName);
    }

    std::ifstream in;
    for (const auto& candidate : candidates) {
        in.open(candidate, std::ios::binary);
        if (in.is_open()) {
            lut.assign(LUT_SIZE, 0.0f);
            in.read(reinterpret_cast<char*>(lut.data()), LUT_SIZE * sizeof(float));
            if (in.gcount() == static_cast<std::streamsize>(LUT_SIZE * sizeof(float))) {
                return true;
            }
            std::cerr << "Gamma LUT file did not contain " << LUT_SIZE
                      << " float32 entries: " << candidate << std::endl;
            return false;
        }
        in.clear();
    }

    std::cerr << "Failed to locate gamma LUT file.";
    if (!candidates.empty()) {
        std::cerr << " Checked: ";
        for (size_t i = 0; i < candidates.size(); ++i) {
            std::cerr << candidates[i];
            if (i + 1 < candidates.size()) std::cerr << ", ";
        }
    }
    std::cerr << std::endl;
    return false;
}

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

__global__ void ep_gpu_optimized_calc_kernel(const uchar3* __restrict__ INPUT,
                                             epPixel* __restrict__ RET,
                                             int width, int height) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < height && j < width) {
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
}

__global__ void ep_gpu_optimized_compare_kernel(const epPixel* __restrict__ prev,
                                                epPixel* __restrict__ curr,
                                                int width, int height,
                                                int* __restrict__ harmfulLumCount,
                                                int* __restrict__ harmfulColCount,
                                                bool HDR) {
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

    // Reduction within the warp respecting the active mask.
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid & (warpSize - 1);
    int warpLum = warpReduceSum(localLum, 0xffffffff);
    int warpCol = warpReduceSum(localCol, 0xffffffff);
    if (lane == 0 && activeMask != 0) {
        if (warpLum) atomicAdd(harmfulLumCount, warpLum);
        if (warpCol) atomicAdd(harmfulColCount, warpCol);
    }
}

} // namespace

bool parse_gpu_optimized_args(int argc, char** argv, GpuOptimizedArgs& out,
                            bool detailRequired) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-in") == 0 && i + 1 < argc) {
            out.input_path = argv[++i];
        } else if (std::strcmp(argv[i], "-out") == 0 && i + 1 < argc) {
            out.output_csv = argv[++i];
        } else if (std::strcmp(argv[i], "-detail") == 0 && i + 1 < argc) {
            out.detail_csv = argv[++i];
        } else if (std::strcmp(argv[i], "-lut") == 0 && i + 1 < argc) {
            out.lut_path = argv[++i];
        } else if (std::strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            out.height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            out.width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-size") == 0 && i + 1 < argc) {
            out.screen_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            out.viewing_distance = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-HDR") == 0) {
            out.hdr = true;
        } else if (std::strcmp(argv[i], "-profile") == 0) {
            out.profile = true;
        }
    }

    const bool requiredPresent =
        !out.input_path.empty() &&
        out.height > 0 &&
        out.width > 0 &&
        out.screen_size != 0 &&
        out.viewing_distance != 0;

    if (!requiredPresent) {
        return false;
    }

    if (detailRequired && out.detail_csv.empty()) {
        return false;
    }

    return true;
}

int run_gpu_optimized_pipeline(const GpuOptimizedArgs& cfg) {
    cv::VideoCapture cap(cfg.input_path);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: cannot open video: " << cfg.input_path << std::endl;
        return 2;
    }

    const int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    const int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int video_length = (fps > 0) ? frame_count / fps : 0;
    const size_t fps_window = static_cast<size_t>(std::max(fps, 1));

    std::ofstream detailStream;
    if (!cfg.detail_csv.empty()) {
        detailStream.open(cfg.detail_csv, std::ios::out);
        if (!detailStream.is_open()) {
            std::cerr << "Failed to open detail output file: "
                      << cfg.detail_csv << std::endl;
            return 3;
        }
        detailStream << "frame,harmfulLumCount,harmfulColCount\n";
    }

    std::vector<float> gammaLUT;
    if (!loadGammaLUT(cfg.lut_path, gammaLUT)) {
        return 4;
    }

    const int W = cfg.width;
    const int H = cfg.height;
    const size_t Npix = static_cast<size_t>(W) * H;
    const size_t BYTES_RGB = Npix * sizeof(uchar3);

    double pixelDensity = computePixelDensity(W, H, cfg.screen_size);
    const double minSafeAreaD =
        computeMinSafeArea(H, W, cfg.viewing_distance, pixelDensity);

    int status = 0;
    int decoderErrFinal = 0;
    std::vector<uchar3*> hostBuffers(HOST_BUFFER_COUNT, nullptr);
    uchar3* dINPUT[2] = {nullptr, nullptr};
    epPixel* dRET[2] = {nullptr, nullptr};
    int *d_harmLum = nullptr, *d_harmCol = nullptr;

    dim3 block;
    dim3 grid;
    std::queue<std::pair<int, int>> oneSecondCounts;
    int freqLum = 0;
    int freqCol = 0;
    bool hasFlash = false;
    bool hasRed = false;
    int processedFrames = 0;
    int prevIdx = 0;
    int curIdx = 1;
    auto start = std::chrono::high_resolution_clock::now();

    const bool profile = cfg.profile;
    std::chrono::duration<double, std::milli> totalDecode{0.0};
    std::chrono::duration<double, std::milli> totalStage{0.0};
    double totalCopyMs = 0.0;
    double totalKernelMs = 0.0;
    cudaEvent_t evtCopyStart = nullptr;
    cudaEvent_t evtCopyEnd = nullptr;
    cudaEvent_t evtKernelStart = nullptr;
    cudaEvent_t evtKernelEnd = nullptr;
    cudaStream_t streams[2] = {nullptr, nullptr};

    struct ReadyFrame {
        int hostIndex = -1;
        double decodeMs = 0.0;
        double stageMs = 0.0;
        bool endOfStream = false;
    };

    std::mutex freeMutex;
    std::condition_variable freeCv;
    std::queue<int> freeHostSlots;

    std::mutex readyMutex;
    std::condition_variable readyCv;
    std::queue<ReadyFrame> readyFrames;

    std::atomic<bool> stopDecoding{false};
    std::atomic<int> decodeStatus{0};
    std::thread decodeThread;
    bool decoderStarted = false;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t __err = (expr);                                             \
        if (__err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(__err)            \
                      << " (" << cudaGetErrorName(__err) << ") at "             \
                      << __FILE__ << ":" << __LINE__ << std::endl;              \
            status = 5;                                                         \
            goto cleanup;                                                       \
        }                                                                       \
    } while (0)

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking));
    if (profile) {
        CUDA_CHECK(cudaEventCreate(&evtCopyStart));
        CUDA_CHECK(cudaEventCreate(&evtCopyEnd));
        CUDA_CHECK(cudaEventCreate(&evtKernelStart));
        CUDA_CHECK(cudaEventCreate(&evtKernelEnd));
    }
    CUDA_CHECK(cudaMemcpyToSymbol(cGammaLUT, gammaLUT.data(),
                                  LUT_SIZE * sizeof(float)));

    for (int i = 0; i < HOST_BUFFER_COUNT; ++i) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostBuffers[i]),
                                 BYTES_RGB, cudaHostAllocPortable));
    }

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dINPUT[i]), BYTES_RGB));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dRET[i]),
                              Npix * sizeof(epPixel)));
        CUDA_CHECK(cudaMemset(dRET[i], 0, Npix * sizeof(epPixel)));
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_harmLum), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_harmCol), sizeof(int)));

    block = dim3(16, 16);
    grid = dim3((W + block.x - 1) / block.x,
                (H + block.y - 1) / block.y);

    for (int i = 0; i < HOST_BUFFER_COUNT; ++i) {
        freeHostSlots.push(i);
    }

    // Background decoder prefetches frames into pinned host buffers.
    decodeThread = std::thread([&, W, H]() {
        cv::Mat frameLocal;
        bool sentEnd = false;

        auto returnSlot = [&](int idx) {
            if (idx >= 0) {
                std::lock_guard<std::mutex> lk(freeMutex);
                freeHostSlots.push(idx);
            }
            freeCv.notify_one();
        };

        auto emitEnd = [&]() {
            if (sentEnd) {
                return;
            }
            ReadyFrame term;
            term.endOfStream = true;
            {
                std::lock_guard<std::mutex> lk(readyMutex);
                readyFrames.push(term);
            }
            readyCv.notify_all();
            sentEnd = true;
        };

        while (!stopDecoding.load(std::memory_order_acquire)) {
            int hostIdx = -1;
            {
                std::unique_lock<std::mutex> lk(freeMutex);
                freeCv.wait(lk, [&] {
                    return stopDecoding.load(std::memory_order_acquire) || !freeHostSlots.empty();
                });
                if (stopDecoding.load(std::memory_order_acquire)) {
                    break;
                }
                hostIdx = freeHostSlots.front();
                freeHostSlots.pop();
            }

            auto decodeStart = std::chrono::high_resolution_clock::now();
            if (!cap.read(frameLocal)) {
                returnSlot(hostIdx);
                emitEnd();
                break;
            }
            auto decodeEnd = std::chrono::high_resolution_clock::now();

            if (frameLocal.channels() != 3) {
                if (frameLocal.channels() == 4) {
                    cv::cvtColor(frameLocal, frameLocal, cv::COLOR_BGRA2BGR);
                } else {
                    decodeStatus.store(7, std::memory_order_release);
                    returnSlot(hostIdx);
                    emitEnd();
                    break;
                }
            }

            if (!frameLocal.isContinuous()) {
                frameLocal = frameLocal.clone();
            }

            const size_t srcStridePixels = frameLocal.step / sizeof(uchar3);
            const size_t srcRows = static_cast<size_t>(frameLocal.rows);
            const size_t maxNeededIndex =
                static_cast<size_t>(H - 1) * srcStridePixels + static_cast<size_t>(W);
            if (srcStridePixels == 0 || srcRows * srcStridePixels < maxNeededIndex) {
                decodeStatus.store(7, std::memory_order_release);
                returnSlot(hostIdx);
                emitEnd();
                break;
            }

            auto stageStart = std::chrono::high_resolution_clock::now();
            const uchar3* srcBase = reinterpret_cast<const uchar3*>(frameLocal.data);
            uchar3* dstBase = hostBuffers[hostIdx];
            for (int row = 0; row < H; ++row) {
                const uchar3* srcRow =
                    srcBase + static_cast<size_t>(row) * srcStridePixels;
                uchar3* dstRow =
                    dstBase + static_cast<size_t>(row) * W;
                std::memcpy(dstRow, srcRow, static_cast<size_t>(W) * sizeof(uchar3));
            }
            auto stageEnd = std::chrono::high_resolution_clock::now();

            ReadyFrame ready;
            ready.hostIndex = hostIdx;
            ready.decodeMs =
                std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
            ready.stageMs =
                std::chrono::duration<double, std::milli>(stageEnd - stageStart).count();

            {
                std::lock_guard<std::mutex> lk(readyMutex);
                readyFrames.push(ready);
            }
            readyCv.notify_one();
        }

        emitEnd();
    });
    decoderStarted = true;

    // Main processing loop consumes staged frames and runs the GPU pipeline.
    while (true) {
        ReadyFrame ready;
        {
            std::unique_lock<std::mutex> lk(readyMutex);
            readyCv.wait(lk, [&] { return !readyFrames.empty(); });
            ready = readyFrames.front();
            readyFrames.pop();
        }

        if (ready.endOfStream) {
            int decoderErr = decodeStatus.load(std::memory_order_acquire);
            if (decoderErr != 0 && status == 0) {
                status = decoderErr;
            }
            break;
        }

        cudaStream_t stream = streams[curIdx];

        if (profile) {
            totalDecode += std::chrono::duration<double, std::milli>(ready.decodeMs);
            totalStage += std::chrono::duration<double, std::milli>(ready.stageMs);
            CUDA_CHECK(cudaEventRecord(evtCopyStart, stream));
        }

        CUDA_CHECK(cudaMemsetAsync(d_harmLum, 0, sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_harmCol, 0, sizeof(int), stream));

        CUDA_CHECK(cudaMemcpyAsync(dINPUT[curIdx], hostBuffers[ready.hostIndex], BYTES_RGB,
                                   cudaMemcpyHostToDevice, stream));

        if (profile) {
            CUDA_CHECK(cudaEventRecord(evtCopyEnd, stream));
            CUDA_CHECK(cudaEventRecord(evtKernelStart, stream));
        }

        ep_gpu_optimized_calc_kernel<<<grid, block, 0, stream>>>(dINPUT[curIdx], dRET[curIdx],
                                                                 W, H);
        CUDA_CHECK(cudaGetLastError());

        ep_gpu_optimized_compare_kernel<<<grid, block, 0, stream>>>(
            dRET[prevIdx], dRET[curIdx],
            W, H,
            d_harmLum, d_harmCol,
            cfg.hdr);
        CUDA_CHECK(cudaGetLastError());

        if (profile) {
            CUDA_CHECK(cudaEventRecord(evtKernelEnd, stream));
            CUDA_CHECK(cudaEventSynchronize(evtCopyEnd));
            float copyElapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&copyElapsed, evtCopyStart, evtCopyEnd));
            totalCopyMs += static_cast<double>(copyElapsed);
            CUDA_CHECK(cudaEventSynchronize(evtKernelEnd));
            float kernelElapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&kernelElapsed, evtKernelStart, evtKernelEnd));
            totalKernelMs += static_cast<double>(kernelElapsed);
        }

        int harmfulLumHost = 0;
        int harmfulColHost = 0;
        CUDA_CHECK(cudaMemcpyAsync(&harmfulLumHost, d_harmLum, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&harmfulColHost, d_harmCol, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        {
            std::lock_guard<std::mutex> lk(freeMutex);
            freeHostSlots.push(ready.hostIndex);
        }
        freeCv.notify_one();

        ++processedFrames;

        if (!cfg.detail_csv.empty() && processedFrames > 1) {
            detailStream << (processedFrames - 1) << ","
                         << harmfulLumHost << ","
                         << harmfulColHost << "\n";
        }

        if (harmfulLumHost > minSafeAreaD) {
            ++freqLum;
        }
        if (harmfulColHost > minSafeAreaD) {
            ++freqCol;
        }

        oneSecondCounts.emplace(harmfulLumHost, harmfulColHost);
        if (oneSecondCounts.size() > fps_window) {
            const auto [oldLum, oldCol] = oneSecondCounts.front();
            if (oldLum > minSafeAreaD) {
                --freqLum;
            }
            if (oldCol > minSafeAreaD) {
                --freqCol;
            }
            oneSecondCounts.pop();
        }

        if (freqLum > 3) {
            hasFlash = true;
        }
        if (freqCol > 3) {
            hasRed = true;
        }

        if (decodeStatus.load(std::memory_order_acquire) != 0 || (hasFlash && hasRed)) {
            stopDecoding.store(true, std::memory_order_release);
            readyCv.notify_all();
            freeCv.notify_all();
            break;
        }

        std::swap(prevIdx, curIdx);
    }

    stopDecoding.store(true, std::memory_order_release);
    readyCv.notify_all();
    freeCv.notify_all();

    decoderErrFinal = decodeStatus.load(std::memory_order_acquire);
    if (decoderErrFinal != 0 && status == 0) {
        status = decoderErrFinal;
    }
    if (status != 0) {
        goto cleanup;
    }

    {
        FILE* outfile = std::fopen(cfg.output_csv.c_str(), "a");
        if (!outfile) {
            std::cerr << "Failed to open summary output file: "
                      << cfg.output_csv << std::endl;
            status = 6;
            goto cleanup;
        }
        const std::string videoName =
            fs::path(cfg.input_path).filename().string();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
        std::fprintf(outfile, "%s,%d,%d,%ld,%d,%d\n",
                     videoName.c_str(),
                     hasFlash ? 1 : 0,
                     hasRed ? 1 : 0,
                     static_cast<long>(duration.count()),
                     video_length,
                     frame_count);
        std::fclose(outfile);
    }

    if (profile && processedFrames > 0) {
        const double frames = static_cast<double>(processedFrames);
        std::cout << "[PROFILE] frames=" << processedFrames
                  << " decode_total_ms=" << totalDecode.count()
                  << " stage_total_ms=" << totalStage.count()
                  << " h2d_total_ms=" << totalCopyMs
                  << " kernel_total_ms=" << totalKernelMs << '\n';
        std::cout << "[PROFILE] per_frame decode_ms=" << (totalDecode.count() / frames)
                  << " stage_ms=" << (totalStage.count() / frames)
                  << " h2d_ms=" << (totalCopyMs / frames)
                  << " kernel_ms=" << (totalKernelMs / frames) << '\n';
    }

cleanup:
    stopDecoding.store(true, std::memory_order_release);
    readyCv.notify_all();
    freeCv.notify_all();
    if (decoderStarted && decodeThread.joinable()) {
        decodeThread.join();
    }

    if (detailStream.is_open()) {
        detailStream.close();
    }

    for (int i = 0; i < 2; ++i) {
        if (dRET[i]) {
            cudaFree(dRET[i]);
        }
        if (dINPUT[i]) {
            cudaFree(dINPUT[i]);
        }
    }

    for (auto* ptr : hostBuffers) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }

    if (d_harmLum) cudaFree(d_harmLum);
    if (d_harmCol) cudaFree(d_harmCol);
    if (streams[0]) cudaStreamDestroy(streams[0]);
    if (streams[1]) cudaStreamDestroy(streams[1]);

    if (profile) {
        cudaEventDestroy(evtCopyStart);
        cudaEventDestroy(evtCopyEnd);
        cudaEventDestroy(evtKernelStart);
        cudaEventDestroy(evtKernelEnd);
    }

#undef CUDA_CHECK

    return status;
}
