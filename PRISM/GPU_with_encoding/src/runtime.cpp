#include "epicap_gpu_runtime.hpp"

#include "gpu_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../../ep_types_gpu.h"

using epicap::launch_calc_kernel;
using epicap::launch_compare_kernel;
using epicap::uploadGammaLUT;

namespace {

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t __err = (expr);                                            \
        if (__err != cudaSuccess) {                                            \
            std::ostringstream __oss;                                          \
            __oss << "CUDA error: " << cudaGetErrorString(__err)               \
                  << " (" << cudaGetErrorName(__err) << ") at "                \
                  << __FILE__ << ":" << __LINE__;                              \
            throw std::runtime_error(__oss.str());                            \
        }                                                                      \
    } while (0)

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

std::vector<float> loadGammaLUT() {
    namespace fs = std::filesystem;
    const fs::path defaultName{"inverseGammaLUT.bin"};
    std::vector<fs::path> candidates = {
        defaultName,
        fs::path("../") / defaultName,
        fs::path("../../") / defaultName,
        fs::path("../../../") / defaultName,
        fs::path(__FILE__).parent_path().parent_path().parent_path() / defaultName
    };

    for (const auto& path : candidates) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            continue;
        }
        std::vector<float> lut(256, 0.0f);
        in.read(reinterpret_cast<char*>(lut.data()), lut.size() * sizeof(float));
        if (in.gcount() == static_cast<std::streamsize>(lut.size() * sizeof(float))) {
            return lut;
        }
    }

    throw std::runtime_error("Failed to load inverse gamma LUT.");
}

class EpicapGpuRuntimeImpl : public ::EpicapGpuRuntime {
public:
    explicit EpicapGpuRuntimeImpl(const EpicapConfig& cfg)
        : cfg_(cfg),
          numPixels_(static_cast<size_t>(cfg.width) * cfg.height),
          rowBytesPacked_(static_cast<size_t>(cfg.width) * sizeof(uchar3)),
          fpsWindow_(static_cast<size_t>(std::max(cfg.fps, 1))),
          detailPath_("") {

        double pixelDensity = computePixelDensity(cfg.width, cfg.height, cfg.screen_size);
        minSafeArea_ = computeMinSafeArea(cfg.height, cfg.width, cfg.viewing_distance, pixelDensity);

        auto lut = loadGammaLUT();
        uploadGammaLUT(lut.data());

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dInput_), rowBytesPacked_ * cfg.height));
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPixels_[i]), numPixels_ * sizeof(epPixel)));
            CUDA_CHECK(cudaMemset(dPixels_[i], 0, numPixels_ * sizeof(epPixel)));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dHarmLum_), sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dHarmCol_), sizeof(int)));
        CUDA_CHECK(cudaMemset(dHarmLum_, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(dHarmCol_, 0, sizeof(int)));
    }

    ~EpicapGpuRuntimeImpl() override {
        cudaFree(dInput_);
        for (int i = 0; i < 2; ++i) {
            cudaFree(dPixels_[i]);
        }
        cudaFree(dHarmLum_);
        cudaFree(dHarmCol_);
        cudaStreamDestroy(stream_);
    }

    void SetDetailOutput(const std::string& detailCsvPath) override {
        detailPath_ = detailCsvPath;
        detailRows_.clear();
    }

    void ProcessFrame(const uint8_t* frameData,
                      size_t frameStrideBytes,
                      EpicapFrameResult& outResult) override {
        if (!frameData) {
            throw std::invalid_argument("frameData cannot be null");
        }
        const size_t srcPitch = frameStrideBytes ? frameStrideBytes : cfg_.width * 3;
        if (srcPitch < cfg_.width * 3) {
            throw std::invalid_argument("Provided frame stride is smaller than row width");
        }

        CUDA_CHECK(cudaMemcpy2DAsync(
            dInput_,
            rowBytesPacked_,
            frameData,
            srcPitch,
            rowBytesPacked_,
            cfg_.height,
            cudaMemcpyHostToDevice,
            stream_));

        processDeviceFrameInternal(reinterpret_cast<const uchar3*>(dInput_), outResult);
    }

    void ProcessFrameDevice(const uint8_t* deviceFrame,
                            size_t frameStrideBytes,
                            EpicapFrameResult& outResult) override {
        if (!deviceFrame) {
            throw std::invalid_argument("deviceFrame cannot be null");
        }
        const size_t srcPitch = frameStrideBytes ? frameStrideBytes : cfg_.width * 3;
        if (srcPitch < cfg_.width * 3) {
            throw std::invalid_argument("Provided frame stride is smaller than row width");
        }

        const uchar3* srcPtr = reinterpret_cast<const uchar3*>(deviceFrame);
        if (srcPitch != rowBytesPacked_) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                dInput_,
                rowBytesPacked_,
                deviceFrame,
                srcPitch,
                rowBytesPacked_,
                cfg_.height,
                cudaMemcpyDeviceToDevice,
                stream_));
            srcPtr = reinterpret_cast<const uchar3*>(dInput_);
        }

        processDeviceFrameInternal(srcPtr, outResult);
    }

    EpicapRuntimeSummary Finalize() override {
        EpicapRuntimeSummary summary{};
        summary.hasFlash = hasFlash_;
        summary.hasRed = hasRed_;
        summary.processedFrames = processedFrames_;
        summary.videoLengthSeconds = cfg_.fps > 0 ? processedFrames_ / cfg_.fps : 0;
        summary.detectionMs = totalDetectionMs_;
        if (!detailPath_.empty()) {
            std::ofstream detail(detailPath_, std::ios::out | std::ios::trunc);
            if (!detail.is_open()) {
                throw std::runtime_error("Failed to open detail CSV: " + detailPath_);
            }
            detail << "frame,harmfulLumCount,harmfulColCount\n";
            for (const auto& row : detailRows_) {
                detail << std::get<0>(row) << ","
                       << std::get<1>(row) << ","
                       << std::get<2>(row) << "\n";
            }
        }
        return summary;
    }

    void Reset() override {
        processedFrames_ = 0;
        freqLum_ = 0;
        freqCol_ = 0;
        hasFlash_ = false;
        hasRed_ = false;
        std::queue<std::pair<int, int>> empty;
        std::swap(oneSecondCounts_, empty);
        detailRows_.clear();
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaMemset(dPixels_[i], 0, numPixels_ * sizeof(epPixel)));
        }
        CUDA_CHECK(cudaMemset(dHarmLum_, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(dHarmCol_, 0, sizeof(int)));
        totalDetectionMs_ = 0.0;
    }

private:
    void processDeviceFrameInternal(const uchar3* deviceFrame,
                                    EpicapFrameResult& outResult) {
        if (!deviceFrame) {
            throw std::invalid_argument("deviceFrame cannot be null");
        }

        launch_calc_kernel(deviceFrame, dPixels_[curIdx_], cfg_.width, cfg_.height, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        CUDA_CHECK(cudaMemsetAsync(dHarmLum_, 0, sizeof(int), stream_));
        CUDA_CHECK(cudaMemsetAsync(dHarmCol_, 0, sizeof(int), stream_));

        auto detStart = std::chrono::high_resolution_clock::now();
        launch_compare_kernel(
            dPixels_[prevIdx_],
            dPixels_[curIdx_],
            cfg_.width,
            cfg_.height,
            cfg_.hdr,
            dHarmLum_,
            dHarmCol_,
            stream_);

        int harmfulLumHost = 0;
        int harmfulColHost = 0;
        CUDA_CHECK(cudaMemcpyAsync(&harmfulLumHost, dHarmLum_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(&harmfulColHost, dHarmCol_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        auto detEnd = std::chrono::high_resolution_clock::now();
        totalDetectionMs_ += std::chrono::duration<double, std::milli>(detEnd - detStart).count();

        ++processedFrames_;
        if (processedFrames_ > 1) {
            recordDetailRow(processedFrames_ - 1, harmfulLumHost, harmfulColHost);
        }

        updateSlidingWindow(harmfulLumHost, harmfulColHost);

        outResult.harmfulLumCount = harmfulLumHost;
        outResult.harmfulColCount = harmfulColHost;
        outResult.hasFlashWindow = freqLum_ > 3;
        outResult.hasRedWindow = freqCol_ > 3;

        std::swap(prevIdx_, curIdx_);
    }

    void updateSlidingWindow(int harmfulLum, int harmfulCol) {
        if (harmfulLum > minSafeArea_) {
            ++freqLum_;
        }
        if (harmfulCol > minSafeArea_) {
            ++freqCol_;
        }

        oneSecondCounts_.emplace(harmfulLum, harmfulCol);
        if (oneSecondCounts_.size() > fpsWindow_) {
            const auto [oldLum, oldCol] = oneSecondCounts_.front();
            if (oldLum > minSafeArea_) {
                --freqLum_;
            }
            if (oldCol > minSafeArea_) {
                --freqCol_;
            }
            oneSecondCounts_.pop();
        }

        if (freqLum_ > 3) {
            hasFlash_ = true;
        }
        if (freqCol_ > 3) {
            hasRed_ = true;
        }
    }

    void recordDetailRow(int frameIndex, int harmfulLum, int harmfulCol) {
        detailRows_.emplace_back(frameIndex, harmfulLum, harmfulCol);
    }

    EpicapConfig cfg_;
    size_t numPixels_;
    size_t rowBytesPacked_;
    size_t fpsWindow_;
    double minSafeArea_;

    cudaStream_t stream_{};
    uchar3* dInput_{nullptr};
    epPixel* dPixels_[2]{nullptr, nullptr};
    int* dHarmLum_{nullptr};
    int* dHarmCol_{nullptr};

    int processedFrames_ = 0;
    int freqLum_ = 0;
    int freqCol_ = 0;
    bool hasFlash_ = false;
    bool hasRed_ = false;
    std::queue<std::pair<int, int>> oneSecondCounts_;

    std::string detailPath_;
    std::vector<std::tuple<int, int, int>> detailRows_;
    int prevIdx_ = 0;
    int curIdx_ = 1;
    double totalDetectionMs_ = 0.0;
};

}  // namespace

std::unique_ptr<EpicapGpuRuntime> EpicapGpuRuntime::Create(const EpicapConfig& cfg) {
    if (cfg.width <= 0 || cfg.height <= 0) {
        throw std::invalid_argument("Invalid resolution for EpicapGpuRuntime::Create");
    }
    if (cfg.fps <= 0) {
        throw std::invalid_argument("FPS must be positive for EpicapGpuRuntime::Create");
    }
    return std::make_unique<EpicapGpuRuntimeImpl>(cfg);
}
