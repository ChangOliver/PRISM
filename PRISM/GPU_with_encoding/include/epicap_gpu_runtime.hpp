#ifndef EPICAP_GPU_RUNTIME_HPP
#define EPICAP_GPU_RUNTIME_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct EpicapConfig {
    int width;
    int height;
    int screen_size;      // inches; -1 to disable
    int viewing_distance; // centimeters or preferred unit; -1 to disable
    int fps;              // frames per second
    bool hdr;
};

struct EpicapFrameResult {
    int harmfulLumCount;
    int harmfulColCount;
    bool hasFlashWindow;  // frequency > 3 within the last second
    bool hasRedWindow;
};

struct EpicapRuntimeSummary {
    bool hasFlash = false;
    bool hasRed = false;
    int processedFrames = 0;
    int videoLengthSeconds = 0;
    double detectionMs = 0.0;
};

class EpicapGpuRuntime {
public:
    static std::unique_ptr<EpicapGpuRuntime> Create(const EpicapConfig& cfg);

    virtual ~EpicapGpuRuntime() = default;

    virtual void SetDetailOutput(const std::string& detailCsvPath) = 0;

    // Submit a frame in BGR interleaved order (OpenCV-compatible uchar3).
    // The caller must ensure `frameData` spans width*height*3 bytes.
    virtual void ProcessFrame(const uint8_t* frameData,
                              size_t frameStrideBytes,
                              EpicapFrameResult& outResult) = 0;

    // Submit a frame already resident on the device.
    virtual void ProcessFrameDevice(const uint8_t* deviceFrame,
                                    size_t frameStrideBytes,
                                    EpicapFrameResult& outResult) = 0;

    // Finalize the stream, optionally recording a summary CSV row.
    virtual EpicapRuntimeSummary Finalize() = 0;

    // Reset internal state for reuse on another stream with same resolution.
    virtual void Reset() = 0;
};

#endif // EPICAP_GPU_RUNTIME_HPP
