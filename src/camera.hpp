#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "config.hpp"

namespace olc {

// Pixel layout of the frames Camera::read() hands back. NV12 is the format the
// Pi camera ISP produces natively; keeping frames in it lets the GPU do the
// YUV->RGB conversion at display time instead of burning a CPU core on it.
enum class PixelFormat { BGR, NV12 };

// A single camera source (Pi camera module or USB webcam).
//
// Frames come back in the source's *native* layout (see format()): the Pi
// camera delivers NV12 straight from the ISP with no CPU colour conversion,
// while a USB webcam is decoded to BGR by OpenCV. Digital zoom is no longer
// baked into the pixels here -- the preview applies it on the GPU (a source
// rect, see zoomSrcRect()), and capture/record materialise a zoomed BGR frame
// on demand via toDisplayBGR(). This keeps the hot preview path free of any
// per-frame CPU colour-convert or resize.
class Camera {
public:
    // Opens the requested source. On CameraKind::Auto it tries the Pi camera
    // (libcamera via GStreamer) first, then falls back to a V4L2 webcam.
    // Returns nullptr if nothing could be opened.
    static std::unique_ptr<Camera> open(const Config& cfg);

    // Grab the next frame in the source's native format (see format()). No zoom
    // is applied. False if the stream ended.
    bool read(cv::Mat& frame);

    // Pixel layout of frames from read().
    PixelFormat format() const { return format_; }

    // The centred crop rectangle for the current zoom, in display-pixel coords.
    // Used as a GPU source rect for the preview, and to crop captures. Returns
    // the full frame when not zoomed. Coordinates are kept even so the rect is
    // valid for chroma-subsampled (NV12) buffers too.
    cv::Rect zoomSrcRect(int w, int h) const;

    // Convert a native frame to a full-size BGR frame, *without* zoom. Used as
    // the first step for capture/record, which need real BGR pixels.
    cv::Mat nativeToBGR(const cv::Mat& native) const;

    // Apply the current digital zoom to a BGR frame in place (centred crop
    // scaled back to full size). No-op when not zoomed.
    void cropZoom(cv::Mat& bgr) const;

    // Materialise a full-size, zoomed BGR frame from a native frame. Equivalent
    // to nativeToBGR() followed by cropZoom(). Not on the preview hot path.
    cv::Mat toDisplayBGR(const cv::Mat& native) const;

    // Crop an even-aligned region out of an NV12 buffer and convert just that
    // region to BGR (no full-frame conversion). `nv12` is a height*3/2 x width
    // single-channel buffer; `r` is in luma pixels, even on all sides.
    static cv::Mat nv12CropToBGR(const cv::Mat& nv12, const cv::Rect& r);

    // Downscale a whole NV12 frame straight to a small BGR image, for handing
    // to face detection. Converting first and resizing after would cost a
    // full-frame YUV->RGB pass every frame -- exactly what the NV12 preview
    // path exists to avoid -- so the Y and the interleaved UV planes are
    // resized separately and only the small result is converted, keeping the
    // work proportional to `targetW` instead of the capture size.
    static cv::Mat nv12ToBGRScaled(const cv::Mat& nv12, int targetW);

    // Encode a BGR region back into `nv12` at `at` (even coords), writing both
    // the Y and the 2x2-subsampled UV planes. Inverse of nv12CropToBGR().
    static void bgrIntoNV12(const cv::Mat& bgr, cv::Mat& nv12, cv::Point at);

    void zoomIn();
    void zoomOut();
    void setZoom(double z);       // absolute zoom, clamped to [1, maxZoom()]
    double zoom() const { return zoom_; }
    bool zoomed() const { return zoom_ > 1.001; }
    static double maxZoom();

    int width() const { return width_; }   // display (luma) width
    int height() const { return height_; }  // display (luma) height
    double fps() const { return fps_; }
    const std::string& description() const { return desc_; }

    // The concrete source that was actually opened. Never Auto: open() resolves
    // Auto to whichever backend started, so callers (e.g. the runtime toggle)
    // always know which physical camera is live.
    CameraKind kind() const { return kind_; }

private:
    Camera() = default;

    cv::VideoCapture cap_;
    PixelFormat format_ = PixelFormat::BGR;
    CameraKind kind_ = CameraKind::Webcam;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 30.0;
    double zoom_ = 1.0;      // 1.0 .. kMaxZoom
    std::string desc_;
};

} // namespace olc
