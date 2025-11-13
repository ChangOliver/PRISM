#ifndef RECON_ENCODER_HPP
#define RECON_ENCODER_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

struct ReconEncoderOptions {
    int width = 0;
    int height = 0;
    int fps = 0;
    int bitrate = 0;
    int crf = -1;
    std::string preset;
    std::string codecName = "libx264";
};

class ReconEncoder {
public:
    using ReconCallback = std::function<void(const uint8_t* bgrData, int strideBytes, int frameIndex)>;

    static std::unique_ptr<ReconEncoder> Create(const ReconEncoderOptions& opts);

    ~ReconEncoder();

    void EncodeFrame(const uint8_t* bgrData,
                     int strideBytes,
                     ReconCallback onReconstructed);

    void Flush(ReconCallback onReconstructed);

    int Width() const { return opts_.width; }
    int Height() const { return opts_.height; }

private:
    explicit ReconEncoder(const ReconEncoderOptions& opts);

    void initCodec();
    void initFrames();
    void initConversionContexts();
    void encodeInternal(const uint8_t* bgrData,
                        int strideBytes,
                        ReconCallback onReconstructed,
                        bool isFlush);

    ReconEncoderOptions opts_;
    const AVCodec* codec_ = nullptr;
    AVCodecContext* ctx_ = nullptr;
    AVFrame* yuvFrame_ = nullptr;
    AVFrame* reconFrame_ = nullptr;
    AVPacket* packet_ = nullptr;
    SwsContext* bgr2yuv_ = nullptr;
    SwsContext* yuv2bgr_ = nullptr;
    std::vector<uint8_t> reconBgrBuffer_;
    uint8_t* reconDstData_[4]{nullptr, nullptr, nullptr, nullptr};
    int reconDstStride_[4]{0, 0, 0, 0};
    int64_t ptsCounter_ = 0;
    bool flushed_ = false;
};

#endif  // RECON_ENCODER_HPP
