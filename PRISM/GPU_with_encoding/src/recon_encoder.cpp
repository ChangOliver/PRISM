#include "recon_encoder.hpp"

#include <stdexcept>
#include <string>

namespace {

inline void throwOnError(int err, const char* what) {
    if (err < 0) {
        char buf[256];
        av_strerror(err, buf, sizeof(buf));
        throw std::runtime_error(std::string(what) + ": " + buf);
    }
}

}  // namespace

std::unique_ptr<ReconEncoder> ReconEncoder::Create(const ReconEncoderOptions& opts) {
    if (opts.width <= 0 || opts.height <= 0 || opts.fps <= 0) {
        throw std::invalid_argument("Invalid encoder options (width/height/fps must be positive)");
    }
    return std::unique_ptr<ReconEncoder>(new ReconEncoder(opts));
}

ReconEncoder::ReconEncoder(const ReconEncoderOptions& opts)
    : opts_(opts) {
    initCodec();
    initFrames();
    initConversionContexts();
    reconBgrBuffer_.resize(static_cast<size_t>(opts_.width) * opts_.height * 3);
    reconDstData_[0] = reconBgrBuffer_.data();
    reconDstStride_[0] = opts_.width * 3;
}

ReconEncoder::~ReconEncoder() {
    try {
        if (!flushed_) {
            Flush(nullptr);
        }
    } catch (...) {
        // best effort in destructor
    }

    if (bgr2yuv_) sws_freeContext(bgr2yuv_);
    if (yuv2bgr_) sws_freeContext(yuv2bgr_);
    if (packet_) av_packet_free(&packet_);
    if (yuvFrame_) av_frame_free(&yuvFrame_);
    if (reconFrame_) av_frame_free(&reconFrame_);
    if (ctx_) avcodec_free_context(&ctx_);
}

void ReconEncoder::EncodeFrame(const uint8_t* bgrData,
                               int strideBytes,
                               ReconCallback onReconstructed) {
    if (!bgrData) {
        throw std::invalid_argument("ReconEncoder::EncodeFrame received null frame data");
    }
    if (flushed_) {
        throw std::runtime_error("Cannot encode after Flush has been called");
    }
    encodeInternal(bgrData, strideBytes, onReconstructed, false);
}

void ReconEncoder::Flush(ReconCallback onReconstructed) {
    if (flushed_) return;
    encodeInternal(nullptr, 0, onReconstructed, true);
    flushed_ = true;
}

void ReconEncoder::initCodec() {
    codec_ = avcodec_find_encoder_by_name(opts_.codecName.c_str());
    if (!codec_) {
        throw std::runtime_error("Failed to find encoder: " + opts_.codecName);
    }

    ctx_ = avcodec_alloc_context3(codec_);
    if (!ctx_) {
        throw std::runtime_error("Failed to allocate AVCodecContext");
    }

    ctx_->width = opts_.width;
    ctx_->height = opts_.height;
    ctx_->time_base = AVRational{1, opts_.fps};
    ctx_->framerate = AVRational{opts_.fps, 1};
    ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx_->flags |= AV_CODEC_FLAG_RECON_FRAME;
    ctx_->gop_size = opts_.fps * 2;
    ctx_->max_b_frames = 0;
    ctx_->thread_count = 0;  // auto
    if (opts_.bitrate > 0) {
        ctx_->bit_rate = opts_.bitrate;
    }

    AVDictionary* opts = nullptr;
    if (!opts_.preset.empty()) {
        av_dict_set(&opts, "preset", opts_.preset.c_str(), 0);
    }
    if (opts_.crf >= 0) {
        av_dict_set(&opts, "crf", std::to_string(opts_.crf).c_str(), 0);
    }

    int ret = avcodec_open2(ctx_, codec_, &opts);
    if (opts) {
        av_dict_free(&opts);
    }
    throwOnError(ret, "avcodec_open2");

    if (!(ctx_->codec->capabilities & AV_CODEC_CAP_ENCODER_RECON_FRAME)) {
        throw std::runtime_error("Encoder does not support reconstructed frame retrieval");
    }

    packet_ = av_packet_alloc();
    if (!packet_) {
        throw std::runtime_error("Failed to allocate AVPacket");
    }
}

void ReconEncoder::initFrames() {
    yuvFrame_ = av_frame_alloc();
    reconFrame_ = av_frame_alloc();
    if (!yuvFrame_ || !reconFrame_) {
        throw std::runtime_error("Failed to allocate AVFrame");
    }

    yuvFrame_->format = ctx_->pix_fmt;
    yuvFrame_->width = ctx_->width;
    yuvFrame_->height = ctx_->height;
    throwOnError(av_frame_get_buffer(yuvFrame_, 32), "av_frame_get_buffer");
}

void ReconEncoder::initConversionContexts() {
    bgr2yuv_ = sws_getContext(
        opts_.width, opts_.height, AV_PIX_FMT_BGR24,
        opts_.width, opts_.height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!bgr2yuv_) {
        throw std::runtime_error("Failed to create BGR->YUV sws context");
    }

    yuv2bgr_ = nullptr;
}

void ReconEncoder::encodeInternal(const uint8_t* bgrData,
                                  int strideBytes,
                                  ReconCallback onReconstructed,
                                  bool isFlush) {
    if (!isFlush) {
        throwOnError(av_frame_make_writable(yuvFrame_), "av_frame_make_writable");

        const uint8_t* srcData[4] = {bgrData, nullptr, nullptr, nullptr};
        int srcStride[4] = {strideBytes, 0, 0, 0};
        int scaled = sws_scale(
            bgr2yuv_,
            srcData,
            srcStride,
            0,
            opts_.height,
            yuvFrame_->data,
            yuvFrame_->linesize);
        if (scaled != opts_.height) {
            throw std::runtime_error("sws_scale failed during BGR->YUV conversion");
        }
        yuvFrame_->pts = ptsCounter_++;
    }

    int ret = avcodec_send_frame(ctx_, isFlush ? nullptr : yuvFrame_);
    if (ret == AVERROR_EOF) {
        return;
    }
    throwOnError(ret, "avcodec_send_frame");

    auto drainReconFrames = [&](ReconCallback cb) {
        while (true) {
            int fret = avcodec_receive_frame(ctx_, reconFrame_);
            if (fret == AVERROR(EAGAIN) || fret == AVERROR_EOF) {
                break;
            }
            throwOnError(fret, "avcodec_receive_frame");

            const uint8_t* src[4] = {
                reconFrame_->data[0],
                reconFrame_->data[1],
                reconFrame_->data[2],
                reconFrame_->data[3]};
            const int srcStride[4] = {
                reconFrame_->linesize[0],
                reconFrame_->linesize[1],
                reconFrame_->linesize[2],
                reconFrame_->linesize[3]};

            if (!src[0]) {
                throw std::runtime_error("Reconstructed frame missing data (encoder did not populate planes)");
            }

            yuv2bgr_ = sws_getCachedContext(
                yuv2bgr_,
                reconFrame_->width,
                reconFrame_->height,
                static_cast<AVPixelFormat>(reconFrame_->format),
                reconFrame_->width,
                reconFrame_->height,
                AV_PIX_FMT_BGR24,
                SWS_BILINEAR,
                nullptr,
                nullptr,
                nullptr);
            if (!yuv2bgr_) {
                throw std::runtime_error("Failed to create/update YUV->BGR sws context");
            }

            const int dstStride = reconFrame_->width * 3;
            if (reconDstStride_[0] != dstStride ||
                reconBgrBuffer_.size() < static_cast<size_t>(dstStride) * reconFrame_->height) {
                reconDstStride_[0] = dstStride;
                reconBgrBuffer_.resize(static_cast<size_t>(dstStride) * reconFrame_->height);
                reconDstData_[0] = reconBgrBuffer_.data();
            }

            int converted = sws_scale(
                yuv2bgr_,
                src,
                srcStride,
                0,
                reconFrame_->height,
                reconDstData_,
                reconDstStride_);
            if (converted != reconFrame_->height) {
                throw std::runtime_error("sws_scale failed during YUV->BGR conversion");
            }

            if (cb) {
                cb(reconDstData_[0], reconDstStride_[0], static_cast<int>(reconFrame_->pts));
            }
            av_frame_unref(reconFrame_);
        }
    };

    while (true) {
        int pret = avcodec_receive_packet(ctx_, packet_);
        if (pret == AVERROR(EAGAIN) || pret == AVERROR_EOF) {
            break;
        }
        throwOnError(pret, "avcodec_receive_packet");

        drainReconFrames(onReconstructed);
        av_packet_unref(packet_);
    }

    drainReconFrames(onReconstructed);
}
