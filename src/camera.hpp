#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "config.hpp"

namespace olc {

// Pixel layout of the frames Camera::read() hands back. NV12 is the format the
// Pi camera ISP produces natively; keeping frames in it lets the GPU do the
// YUV->RGB conversion at display time instead of burning a CPU core on it.
enum class PixelFormat { BGR, NV12 };

// One camera the app can run on: everything open() needs to start that exact
// device, plus a short label for the on-screen toast. Camera::sources() builds
// the list, so the UI knows how many cameras there are without opening any of
// them, and the switch button is offered exactly when there is a second one.
struct CameraSource {
    CameraKind kind = CameraKind::Webcam; // never Auto: this is a real device
    int index = -1;    // /dev/videoN for a webcam; unused for the Pi camera
    std::string name;  // libcamera camera-name, when one was asked for
    std::string label; // for the UI: "Pi camera", or the webcam's model name

    // Same physical device? Compared on what identifies it, not on the label,
    // which the driver supplies and which is only ever displayed.
    bool operator==(const CameraSource& o) const {
        return kind == o.kind && index == o.index && name == o.name;
    }
};

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
    // Every camera the app may use, in preference order: the Pi camera first
    // (libcamera via GStreamer), then each USB webcam, narrowed by --camera /
    // --webcam-index. Webcams are enumerated from the V4L2 nodes that really
    // exist; the Pi camera is listed without being started, since only opening
    // it reveals whether libcamera has a sensor -- so callers should drop an
    // entry whose open() fails.
    static std::vector<CameraSource> sources(const Config& cfg);

    // Opens one specific source. Returns nullptr if it would not start.
    static std::unique_ptr<Camera> open(const Config& cfg, const CameraSource& src);

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

    // The source this camera was opened from, so the caller can tell which of
    // the available cameras is live (the switch button rotates through them).
    const CameraSource& source() const { return src_; }

private:
    Camera() = default;

    cv::VideoCapture cap_;
    CameraSource src_;
    PixelFormat format_ = PixelFormat::BGR;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 30.0;
    double zoom_ = 1.0;      // 1.0 .. kMaxZoom
    std::string desc_;
};

} // namespace olc
