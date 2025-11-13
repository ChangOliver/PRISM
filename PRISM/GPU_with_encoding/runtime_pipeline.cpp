#include "epicap_gpu_runtime.hpp"
#include "gpu_kernels.hpp"
#include "recon_encoder.hpp"

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t __err = (expr);                                            \
        if (__err != cudaSuccess) {                                            \
            std::ostringstream __oss;                                          \
            __oss << "CUDA error: " << cudaGetErrorString(__err)               \
                  << " (" << cudaGetErrorName(__err) << ") at "                \
                  << __FILE__ << ":" << __LINE__;                              \
            throw std::runtime_error(__oss.str());                             \
        }                                                                      \
    } while (0)

struct VariantConfig {
    std::string name;
    int width;
    int height;
    std::string detailPath;
    int bitrate = 0;
    std::string codecName;
    std::unique_ptr<EpicapGpuRuntime> runtime;
    std::unique_ptr<ReconEncoder> encoder;
    uint8_t* deviceFrame = nullptr;
    size_t deviceStrideBytes = 0;
    std::vector<uint8_t> hostBgr;
    size_t hostStrideBytes = 0;
    double resizeMs = 0.0;
    double encodeMs = 0.0;
    double detectionMs = 0.0;
    bool hasFlash = false;
    bool hasRed = false;
    int processedFrames = 0;
    int videoLengthSeconds = 0;
};

struct PipelineConfig {
    std::string videoPath;
    std::string summaryPath;
    int fps = 0;
    int screenSize = 14;
    int viewingDistance = 24;
    bool hdr = false;
    std::string preset;
    int crf = -1;
    std::string codecName = "libx264";
    std::vector<VariantConfig> variants;
};

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --video <path> --summary <summary_csv> --fps <fps> "
                 "[--size <inches>] [--distance <cm>] [--hdr] "
                 "--variant <name>:<width>:<height>:<detail_csv> [...])\n";
}

static PipelineConfig parseArgs(int argc, char** argv) {
    PipelineConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto requireValue = [&](const char* opt) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + opt);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--video") {
            cfg.videoPath = requireValue("--video");
        } else if (arg == "--summary") {
            cfg.summaryPath = requireValue("--summary");
        } else if (arg == "--fps") {
            cfg.fps = std::stoi(requireValue("--fps"));
        } else if (arg == "--size") {
            cfg.screenSize = std::stoi(requireValue("--size"));
        } else if (arg == "--distance") {
            cfg.viewingDistance = std::stoi(requireValue("--distance"));
        } else if (arg == "--hdr") {
            cfg.hdr = true;
        } else if (arg == "--preset") {
            cfg.preset = requireValue("--preset");
        } else if (arg == "--crf") {
            cfg.crf = std::stoi(requireValue("--crf"));
        } else if (arg == "--codec") {
            cfg.codecName = requireValue("--codec");
        } else if (arg == "--variant") {
            std::string spec = requireValue("--variant");
            VariantConfig vc;
            const size_t pos1 = spec.find(':');
            const size_t pos2 = spec.find(':', pos1 + 1);
            const size_t pos3 = spec.find(':', pos2 + 1);
            const size_t pos4 = spec.find(':', pos3 + 1);
            if (pos1 == std::string::npos || pos2 == std::string::npos ||
                pos3 == std::string::npos || pos4 == std::string::npos) {
                throw std::runtime_error("Invalid variant spec: " + spec);
            }
            vc.name = spec.substr(0, pos1);
            vc.width = std::stoi(spec.substr(pos1 + 1, pos2 - pos1 - 1));
            vc.height = std::stoi(spec.substr(pos2 + 1, pos3 - pos2 - 1));
            vc.bitrate = std::stoi(spec.substr(pos3 + 1, pos4 - pos3 - 1));
            vc.detailPath = spec.substr(pos4 + 1);
            cfg.variants.push_back(std::move(vc));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.videoPath.empty() || cfg.summaryPath.empty() || cfg.variants.empty() || cfg.fps <= 0) {
        throw std::runtime_error("Missing required arguments");
    }
    return cfg;
}

static void prepareVariants(PipelineConfig& cfg) {
    for (auto& variant : cfg.variants) {
        EpicapConfig rtCfg{};
        rtCfg.width = variant.width;
        rtCfg.height = variant.height;
        rtCfg.screen_size = cfg.screenSize;
        rtCfg.viewing_distance = cfg.viewingDistance;
        rtCfg.fps = cfg.fps;
        rtCfg.hdr = cfg.hdr;

        variant.runtime = EpicapGpuRuntime::Create(rtCfg);
        variant.runtime->SetDetailOutput(variant.detailPath);

        variant.deviceStrideBytes = static_cast<size_t>(variant.width) * sizeof(uchar3);
        const size_t allocBytes = variant.deviceStrideBytes * static_cast<size_t>(variant.height);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&variant.deviceFrame), allocBytes));
        variant.hostStrideBytes = static_cast<size_t>(variant.width) * sizeof(uchar3);
        variant.hostBgr.resize(variant.hostStrideBytes * static_cast<size_t>(variant.height));

        variant.codecName = cfg.codecName;
        ReconEncoderOptions encOpts{};
        encOpts.width = variant.width;
        encOpts.height = variant.height;
        encOpts.fps = cfg.fps;
        encOpts.bitrate = variant.bitrate;
        encOpts.preset = cfg.preset;
        encOpts.crf = cfg.crf;
        encOpts.codecName = cfg.codecName.empty() ? "libx264" : cfg.codecName;
        variant.encoder = ReconEncoder::Create(encOpts);

        variant.resizeMs = 0.0;
        variant.encodeMs = 0.0;
        variant.detectionMs = 0.0;
    }
}

static void releaseVariantBuffers(PipelineConfig& cfg) {
    for (auto& variant : cfg.variants) {
        if (variant.deviceFrame) {
            cudaFree(variant.deviceFrame);
            variant.deviceFrame = nullptr;
        }
        variant.encoder.reset();
        variant.hostBgr.clear();
        variant.hostStrideBytes = 0;
    }
}

int main(int argc, char** argv) {
    PipelineConfig cfg;
    cudaStream_t pipelineStream = nullptr;
    uint8_t* decodedDeviceFrame = nullptr;
    size_t decodedStrideBytes = 0;
    int decodedWidth = 0;
    int decodedHeight = 0;

    auto cleanup = [&](bool destroyStream) {
        releaseVariantBuffers(cfg);
        if (decodedDeviceFrame) {
            cudaFree(decodedDeviceFrame);
            decodedDeviceFrame = nullptr;
        }
        if (destroyStream && pipelineStream) {
            cudaStreamDestroy(pipelineStream);
            pipelineStream = nullptr;
        }
    };

    try {
        cfg = parseArgs(argc, argv);
        prepareVariants(cfg);

        CUDA_CHECK(cudaStreamCreateWithFlags(&pipelineStream, cudaStreamNonBlocking));

        fs::path summaryPath(cfg.summaryPath);
        {
            std::ofstream summary(summaryPath, std::ios::out | std::ios::trunc);
            if (!summary.is_open()) {
                throw std::runtime_error("Failed to open summary file: " + summaryPath.string());
            }
        }

        double decodeMs = 0.0;
        const auto pipelineStart = std::chrono::high_resolution_clock::now();

        cv::VideoCapture cap(cfg.videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video: " << cfg.videoPath << std::endl;
            cleanup(true);
            return 2;
        }

        const std::string videoName = fs::path(cfg.videoPath).filename().string();

        cv::Mat frameBGR;
        while (true) {
            auto decodeStart = std::chrono::high_resolution_clock::now();
            if (!cap.read(frameBGR)) {
                break;
            }
            auto decodeEnd = std::chrono::high_resolution_clock::now();
            decodeMs += std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

            if (!frameBGR.isContinuous()) {
                frameBGR = frameBGR.clone();
            }

            if (decodedWidth != frameBGR.cols || decodedHeight != frameBGR.rows) {
                if (decodedDeviceFrame) {
                    cudaFree(decodedDeviceFrame);
                    decodedDeviceFrame = nullptr;
                }
                decodedWidth = frameBGR.cols;
                decodedHeight = frameBGR.rows;
                decodedStrideBytes = static_cast<size_t>(decodedWidth) * sizeof(uchar3);
                const size_t allocBytes = decodedStrideBytes * static_cast<size_t>(decodedHeight);
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&decodedDeviceFrame), allocBytes));
            }

            const size_t uploadBytes = decodedStrideBytes * static_cast<size_t>(decodedHeight);
            bool baseAttributed = false;
            double baseConversionMs = 0.0;
            auto encodeStartGlobal = std::chrono::high_resolution_clock::now();

            if (!cfg.variants.empty()) {
                auto uploadStart = std::chrono::high_resolution_clock::now();
                CUDA_CHECK(cudaMemcpyAsync(
                    decodedDeviceFrame,
                    frameBGR.ptr<uint8_t>(),
                    uploadBytes,
                    cudaMemcpyHostToDevice,
                    pipelineStream));
                CUDA_CHECK(cudaStreamSynchronize(pipelineStream));
                auto uploadEnd = std::chrono::high_resolution_clock::now();
                baseConversionMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
            }

            for (auto& variant : cfg.variants) {
                const bool sameSize = (variant.width == decodedWidth) && (variant.height == decodedHeight);
                const uint8_t* encodeBgr = nullptr;
                size_t strideBytes = 0;

                if (sameSize) {
                    if (!baseAttributed) {
                        variant.resizeMs += baseConversionMs;
                        baseAttributed = true;
                    }
                    encodeBgr = frameBGR.ptr<uint8_t>();
                    strideBytes = static_cast<size_t>(frameBGR.step);
                } else {
                    auto convStart = std::chrono::high_resolution_clock::now();
                    epicap::launch_resize_bilinear(
                        reinterpret_cast<const uchar3*>(decodedDeviceFrame),
                        decodedWidth,
                        decodedHeight,
                        decodedStrideBytes,
                        reinterpret_cast<uchar3*>(variant.deviceFrame),
                        variant.width,
                        variant.height,
                        variant.deviceStrideBytes,
                        pipelineStream);
                    CUDA_CHECK(cudaStreamSynchronize(pipelineStream));
                    CUDA_CHECK(cudaMemcpyAsync(
                        variant.hostBgr.data(),
                        variant.deviceFrame,
                        variant.deviceStrideBytes * static_cast<size_t>(variant.height),
                        cudaMemcpyDeviceToHost,
                        pipelineStream));
                    CUDA_CHECK(cudaStreamSynchronize(pipelineStream));
                    auto convEnd = std::chrono::high_resolution_clock::now();
                    variant.resizeMs += std::chrono::duration<double, std::milli>(convEnd - convStart).count();

                    encodeBgr = variant.hostBgr.data();
                    strideBytes = variant.hostStrideBytes;
                }

                auto reconCallback = [&](const uint8_t* reconBgr, int reconStride, int /*frameIndex*/) {
                    EpicapFrameResult result{};
                    variant.runtime->ProcessFrame(reconBgr, static_cast<size_t>(reconStride), result);
                };

                auto encodeStart = std::chrono::high_resolution_clock::now();
                variant.encoder->EncodeFrame(
                    encodeBgr,
                    static_cast<int>(strideBytes),
                    reconCallback);
                auto encodeEnd = std::chrono::high_resolution_clock::now();
                variant.encodeMs += std::chrono::duration<double, std::milli>(encodeEnd - encodeStart).count();
            }
        }

        for (auto& variant : cfg.variants) {
            auto reconCallback = [&](const uint8_t* reconBgr, int reconStride, int /*frameIndex*/) {
                EpicapFrameResult result{};
                variant.runtime->ProcessFrame(reconBgr, static_cast<size_t>(reconStride), result);
            };
            auto flushStart = std::chrono::high_resolution_clock::now();
            variant.encoder->Flush(reconCallback);
            auto flushEnd = std::chrono::high_resolution_clock::now();
            variant.encodeMs += std::chrono::duration<double, std::milli>(flushEnd - flushStart).count();
        }

        std::ofstream summary(summaryPath, std::ios::out | std::ios::app);
        if (!summary.is_open()) {
            throw std::runtime_error("Failed to open summary file: " + summaryPath.string());
        }
        for (auto& variant : cfg.variants) {
            EpicapRuntimeSummary rs = variant.runtime->Finalize();
            variant.detectionMs = rs.detectionMs;
            variant.hasFlash = rs.hasFlash;
            variant.hasRed = rs.hasRed;
            variant.processedFrames = rs.processedFrames;
            variant.videoLengthSeconds = rs.videoLengthSeconds;
            summary << videoName << ','
                    << variant.name << ','
                    << (variant.hasFlash ? 1 : 0) << ','
                    << (variant.hasRed ? 1 : 0) << ','
                    << variant.videoLengthSeconds << ','
                    << variant.processedFrames << '\n';
        }
        summary.close();

        const fs::path timingsPath = summaryPath.parent_path() / "timings.csv";
        std::ofstream timings(timingsPath, std::ios::out | std::ios::trunc);
        if (!timings.is_open()) {
            throw std::runtime_error("Failed to open timings file: " + timingsPath.string());
        }
        timings << "video,variant,decode_ms,resize_ms,encode_ms,detection_ms\n";
        bool decodeAttributed = false;
        for (const auto& variant : cfg.variants) {
            const double decodeShare = decodeAttributed ? 0.0 : decodeMs;
            decodeAttributed = true;
            timings << videoName << ','
                    << variant.name << ','
                    << decodeShare << ','
                    << variant.resizeMs << ','
                    << variant.encodeMs << ','
                    << variant.detectionMs << '\n';
        }
        std::cout << "[SUCCESS] Timings written to " << timingsPath << std::endl;

        cleanup(true);

    } catch (const std::exception& ex) {
        cleanup(true);
        std::cerr << "Error: " << ex.what() << std::endl;
        usage(argv[0]);
        return 1;
    }

    return 0;
}
