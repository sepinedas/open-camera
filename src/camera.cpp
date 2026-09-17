#include "camera.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

namespace olc {

namespace {
constexpr double kMaxZoom = 4.0;
constexpr double kZoomStep = 1.25; // multiplicative per tap

// Number of CPU threads to hand the software colour converter in the fallback
// pipeline. A single-threaded videoconvert becomes the frame-rate bottleneck
// while the Pi 5's other three Cortex-A76 cores sit idle.
int convertThreads() {
    unsigned n = std::thread::hardware_concurrency();
    return static_cast<int>(n == 0 ? 1 : n);
}

// Fast path: pull NV12 straight from the ISP with *no* colour conversion in the
// pipeline. Combined with CAP_PROP_CONVERT_RGB=0 this hands OpenCV the raw
// planar buffer, which the display then uploads to an NV12 texture so the GPU
// does the YUV->RGB conversion. A short leaky queue decouples capture from the
// consumer and always keeps the freshest frame.
std::string picamPipelineNV12(int w, int h, const std::string& name) {
    std::string src = "libcamerasrc";
    if (!name.empty()) src += " camera-name=\"" + name + "\"";
    return src + " ! video/x-raw,format=NV12" +
           ",width=" + std::to_string(w) + ",height=" + std::to_string(h) +
           " ! queue leaky=downstream max-size-buffers=2"
           " ! appsink drop=true max-buffers=2";
}

// Fallback: ISP-processed format -> BGR via the software converter. Used only
// when the GPU can't take NV12 textures or NV12 capture won't start. The
// converter is spread across all cores (n-threads) and a leaky queue lets
// capture and conversion run on separate threads instead of in lock-step.
//
// The source caps MUST pin a *processed* pixel format (NV12/YUV420/RGBx). If
// they don't, libcamerasrc may negotiate the sensor's native Bayer stream
// (e.g. the IMX500's 2028x1520-SRGGB16/RAW), which videoconvert can't consume,
// and the pipeline fails to start. `format` forces the ISP-processed output.
// `name` optionally selects one camera by its libcamera id when several exist.
std::string picamPipelineBGR(const std::string& format, int w, int h,
                             const std::string& name) {
    std::string src = "libcamerasrc";
    if (!name.empty()) src += " camera-name=\"" + name + "\"";
    return src + " ! video/x-raw,format=" + format +
           ",width=" + std::to_string(w) + ",height=" + std::to_string(h) +
           " ! queue leaky=downstream max-size-buffers=2"
           " ! videoconvert n-threads=" + std::to_string(convertThreads()) +
           " ! video/x-raw,format=BGR"
           " ! appsink drop=true max-buffers=2";
}

// Ask /dev/videoN what it is, without starting a stream. Returns false unless
// the node can actually deliver captured video to userspace, which rules out
// the two kinds of node that are not webcams:
//   * the metadata node every UVC camera exposes next to its video node (it
//     reports V4L2_CAP_META_CAPTURE, not VIDEO_CAPTURE), and
//   * the Pi's own CSI receiver, ISP and codec nodes, which do report video
//     capture but belong to libcamera (or to the hardware encoder) -- opened
//     as a plain webcam they deliver either nothing or raw Bayer frames.
// On success `card` gets the driver's name for the device, which is what the
// switch button shows ("HD Pro Webcam C920").
bool queryCapture(int index, std::string& card) {
    std::string path = "/dev/video" + std::to_string(index);
    int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) return false;
    v4l2_capability cap{};
    bool ok = ::ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0;
    ::close(fd);
    if (!ok) return false;

    // device_caps describes this node; capabilities describes the whole device
    // (all its nodes together), so a metadata node would pass on the latter.
    unsigned caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps
                                                              : cap.capabilities;
    if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING))
        return false;
    // A memory-to-memory node (the hardware codecs) advertises capture *and*
    // output; a camera never outputs.
    if (caps & V4L2_CAP_VIDEO_OUTPUT) return false;

    auto field = [](const __u8* p, size_t n) {
        const char* s = reinterpret_cast<const char*>(p);
        return std::string(s, ::strnlen(s, n));
    };
    static const char* kLibcameraDrivers[] = {"bcm2835-isp", "bcm2835-codec",
                                              "rp1-cfe", "unicam", "pispbe"};
    std::string driver = field(cap.driver, sizeof(cap.driver));
    for (const char* d : kLibcameraDrivers)
        if (driver == d) return false;

    card = field(cap.card, sizeof(cap.card));
    return true;
}

// Short label for a webcam, shown in the toast after a switch. The driver's
// card name is the friendly one ("HD Pro Webcam C920"); the node number is the
// fallback for a driver that leaves it blank.
std::string webcamLabel(int index, const std::string& card) {
    return card.empty() ? "USB camera " + std::to_string(index) : card;
}

// Try one V4L2 index; returns true and leaves `cap` open on success.
bool tryWebcam(cv::VideoCapture& cap, int index, int w, int h) {
    if (!cap.open(index, cv::CAP_V4L2)) return false;
    // Prefer MJPG: even on the Pi 5's USB 3.0 ports, an uncompressed YUYV
    // stream at 1080p30 eats most of the bus, and most webcams only offer
    // their higher resolutions over MJPG anyway.
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    cv::Mat probe;
    if (!cap.read(probe) || probe.empty()) {
        cap.release();
        return false;
    }
    return true;
}
} // namespace

// Enumerate rather than probe: opening ten /dev/video nodes in turn (what the
// webcam fallback used to do) costs a stream start each, far too slow to run
// just to decide whether there is a second camera to offer.
std::vector<CameraSource> Camera::sources(const Config& cfg) {
    std::vector<CameraSource> out;

    // The Pi camera goes first, so --camera auto still prefers it.
    if (cfg.camera != CameraKind::Webcam)
        out.push_back({CameraKind::PiCam, -1, cfg.picamName, "Pi camera"});

    if (cfg.camera != CameraKind::PiCam) {
        if (cfg.webcamIndex >= 0) {
            // Explicitly forced: offer it whatever the node says about itself.
            std::string card;
            queryCapture(cfg.webcamIndex, card);
            out.push_back({CameraKind::Webcam, cfg.webcamIndex, "",
                           webcamLabel(cfg.webcamIndex, card)});
        } else {
            for (int i = 0; i < 64; ++i) {
                std::string card;
                if (queryCapture(i, card))
                    out.push_back({CameraKind::Webcam, i, "",
                                   webcamLabel(i, card)});
            }
        }
    }
    return out;
}

std::unique_ptr<Camera> Camera::open(const Config& cfg, const CameraSource& src) {
    std::unique_ptr<Camera> cam(new Camera());
    cam->src_ = src;

    // Preferred Pi path: raw NV12 delivered to OpenCV untouched, so the GPU can
    // convert it at display time. Ask OpenCV not to auto-convert to BGR; a
    // successful raw grab comes back as a single-channel planar buffer
    // (height*3/2 rows). If the backend ignores that and still hands us BGR
    // (3 channels), NV12 capture isn't available -- fall through to the BGR
    // pipeline below.
    auto openPiNV12 = [&]() -> bool {
        std::string pipe = picamPipelineNV12(cfg.width, cfg.height, src.name);
        if (!cam->cap_.open(pipe, cv::CAP_GSTREAMER)) return false;
        cam->cap_.set(cv::CAP_PROP_CONVERT_RGB, 0);
        cv::Mat probe;
        if (cam->cap_.read(probe) && !probe.empty() && probe.channels() == 1 &&
            probe.rows % 3 == 0) {
            cam->format_ = PixelFormat::NV12;
            cam->width_ = probe.cols;
            cam->height_ = probe.rows * 2 / 3; // strip the interleaved UV plane
            cam->desc_ = "Pi camera (libcamera, NV12 -> GPU convert)";
            return true;
        }
        cam->cap_.release();
        std::cerr << "picam: raw NV12 capture unavailable; using CPU convert\n";
        return false;
    };

    auto openPiBGR = [&]() -> bool {
        // Different sensors/ISPs expose different processed formats; try the
        // common ones in turn and keep the first that actually delivers a frame.
        static const char* kFormats[] = {"NV12", "YUV420", "RGBx", "BGRx", "RGB"};
        for (const char* fmt : kFormats) {
            std::string pipe = picamPipelineBGR(fmt, cfg.width, cfg.height, src.name);
            if (!cam->cap_.open(pipe, cv::CAP_GSTREAMER)) continue;
            cv::Mat probe;
            if (cam->cap_.read(probe) && !probe.empty()) {
                cam->format_ = PixelFormat::BGR;
                cam->desc_ = std::string("Pi camera (libcamera, ") + fmt +
                             " -> CPU convert)";
                return true;
            }
            cam->cap_.release();
            std::cerr << "picam: format " << fmt << " did not start; trying next\n";
        }
        std::cerr << "picam: no libcamera format worked. Check `rpicam-hello` "
                     "and `gst-inspect-1.0 libcamerasrc`.\n";
        return false;
    };

    auto openPi = [&]() -> bool { return openPiNV12() || openPiBGR(); };

    auto openWebcam = [&]() -> bool {
        cam->format_ = PixelFormat::BGR;
        if (src.index < 0) return false;
        if (!tryWebcam(cam->cap_, src.index, cfg.width, cfg.height)) return false;
        cam->desc_ = src.label + " (/dev/video" + std::to_string(src.index) + ")";
        return true;
    };

    // sources() never yields Auto, so the kind names one concrete backend.
    if (!(src.kind == CameraKind::PiCam ? openPi() : openWebcam()))
        return nullptr;

    // For the BGR paths, record the actual negotiated geometry (the NV12 path
    // already set width_/height_ from the raw buffer above).
    if (cam->format_ != PixelFormat::NV12) {
        cam->width_ = static_cast<int>(cam->cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        cam->height_ = static_cast<int>(cam->cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    }
    double fps = cam->cap_.get(cv::CAP_PROP_FPS);
    cam->fps_ = (fps > 1.0 && fps < 121.0) ? fps : 30.0;
    if (cam->width_ <= 0 || cam->height_ <= 0) {
        cam->width_ = cfg.width;
        cam->height_ = cfg.height;
    }
    return cam;
}

void Camera::zoomIn() { zoom_ = std::min(kMaxZoom, zoom_ * kZoomStep); }
void Camera::zoomOut() { zoom_ = std::max(1.0, zoom_ / kZoomStep); }
void Camera::setZoom(double z) { zoom_ = std::min(kMaxZoom, std::max(1.0, z)); }
double Camera::maxZoom() { return kMaxZoom; }

// Centred crop for the current zoom, in display pixels. Kept even on all sides
// so it stays valid for NV12's 2x2-subsampled chroma plane.
cv::Rect Camera::zoomSrcRect(int w, int h) const {
    if (zoom_ <= 1.001 || w <= 0 || h <= 0) return cv::Rect(0, 0, w, h);
    int cw = std::max(2, static_cast<int>(w / zoom_)) & ~1;
    int ch = std::max(2, static_cast<int>(h / zoom_)) & ~1;
    int x = ((w - cw) / 2) & ~1;
    int y = ((h - ch) / 2) & ~1;
    if (x + cw > w) x = w - cw;
    if (y + ch > h) y = h - ch;
    return cv::Rect(x, y, cw, ch);
}

// Convert a native frame to full-size BGR, no zoom. The clone for an
// already-BGR source hands callers their own buffer to mutate (the filters do).
cv::Mat Camera::nativeToBGR(const cv::Mat& native) const {
    cv::Mat bgr;
    if (native.empty()) return bgr;
    if (format_ == PixelFormat::NV12)
        cv::cvtColor(native, bgr, cv::COLOR_YUV2BGR_NV12);
    else
        bgr = native.clone();
    return bgr;
}

// Digital zoom on a BGR frame: crop the centred window and scale it back to the
// full frame. Uniform across backends so a saved photo matches the preview.
void Camera::cropZoom(cv::Mat& bgr) const {
    if (zoom_ <= 1.001 || bgr.empty()) return;
    cv::Rect roi = zoomSrcRect(bgr.cols, bgr.rows);
    cv::Mat zoomed;
    cv::resize(bgr(roi), zoomed, bgr.size(), 0, 0, cv::INTER_LINEAR);
    bgr = zoomed;
}

cv::Mat Camera::toDisplayBGR(const cv::Mat& native) const {
    cv::Mat bgr = nativeToBGR(native);
    cropZoom(bgr);
    return bgr;
}

// Assemble a contiguous NV12 sub-buffer for `r` (Y block + matching UV block),
// then convert it to BGR. Keeps colour conversion proportional to the region
// instead of the whole frame.
cv::Mat Camera::nv12CropToBGR(const cv::Mat& nv12, const cv::Rect& r) {
    const int H = nv12.rows * 2 / 3;
    cv::Mat roi(r.height * 3 / 2, r.width, CV_8UC1);
    // Y plane: rows [r.y, r.y+r.height).
    nv12(cv::Rect(r.x, r.y, r.width, r.height))
        .copyTo(roi(cv::Rect(0, 0, r.width, r.height)));
    // Interleaved UV plane: it lives below the Y plane (starting at row H) at
    // half the vertical resolution; the chroma for luma column x sits at byte
    // column x, so the region's byte columns match the Y region's.
    nv12(cv::Rect(r.x, H + r.y / 2, r.width, r.height / 2))
        .copyTo(roi(cv::Rect(0, r.height, r.width, r.height / 2)));
    cv::Mat bgr;
    cv::cvtColor(roi, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}

cv::Mat Camera::nv12ToBGRScaled(const cv::Mat& nv12, int targetW) {
    if (nv12.empty() || nv12.type() != CV_8UC1 || nv12.rows % 3 != 0)
        return cv::Mat();
    const int W = nv12.cols, H = nv12.rows * 2 / 3;
    if (W < 2 || H < 2 || targetW < 2) return cv::Mat();

    // NV12's 2x2 chroma sampling needs both dimensions even, on the source
    // plane and on the downscaled one.
    int w = std::min(targetW, W) & ~1;
    int h = static_cast<int>(std::lround(static_cast<double>(H) * w / W)) & ~1;
    if (w < 2 || h < 2) return cv::Mat();

    // The plane-wise path below reinterprets the UV bytes as two channels,
    // which needs the rows packed. A capture buffer normally is; if this one
    // is not, fall back to the straightforward (costlier) whole-frame convert.
    if (!nv12.isContinuous() || W % 2 != 0 || H % 2 != 0) {
        cv::Mat full, out;
        cv::cvtColor(nv12, full, cv::COLOR_YUV2BGR_NV12);
        cv::resize(full, out, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        return out;
    }

    cv::Mat small(h * 3 / 2, w, CV_8UC1);
    cv::Mat dstY = small(cv::Rect(0, 0, w, h));
    cv::resize(nv12.rowRange(0, H), dstY, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    // View the interleaved chroma rows as CV_8UC2 so U and V are resized as
    // separate channels; resizing the raw bytes would blend them into mush.
    cv::Mat uv = nv12.rowRange(H, H + H / 2).reshape(2, H / 2);
    cv::Mat uvSmall;
    cv::resize(uv, uvSmall, cv::Size(w / 2, h / 2), 0, 0, cv::INTER_AREA);
    uvSmall.reshape(1, h / 2).copyTo(small(cv::Rect(0, h, w, h / 2)));

    cv::Mat bgr;
    cv::cvtColor(small, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}

// Inverse of nv12CropToBGR: write a filtered BGR region back into the NV12
// buffer, re-deriving the Y plane and the 2x2-averaged UV plane.
void Camera::bgrIntoNV12(const cv::Mat& bgr, cv::Mat& nv12, cv::Point at) {
    if (bgr.empty() || bgr.type() != CV_8UC3) return;
    const int H = nv12.rows * 2 / 3;
    // This has to be the exact inverse of nv12CropToBGR's COLOR_YUV2BGR_NV12,
    // which is *limited range* (Y 16..235, the video convention the camera
    // delivers and the GPU assumes when it converts the whole frame at blit
    // time). COLOR_BGR2YUV is full range, so round tripping through the pair
    // stretched contrast -- shadows up to ~13 levels darker, highlights ~20
    // brighter. Only the filtered region made that trip, so the dirty
    // rectangle stopped matching the pixels around it and showed up as a dark
    // frame around the face.
    //
    // YCrCb rather than YUV: OpenCV's COLOR_BGR2YUV uses the analog 0.492/0.877
    // weights, which would need a *different* rescale per chroma channel
    // (x1.007 and x0.714). YCrCb's 0.564/0.713 weights both come out at
    // 224/255, so one factor serves both and the inverse is exact.
    constexpr double kYScale = 219.0 / 255.0;
    constexpr double kCScale = 224.0 / 255.0;
    cv::Mat ycc;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    cv::Mat ch[3]; // Y, Cr, Cb -- note NV12 interleaves Cb first
    cv::split(ycc, ch);
    ch[0].convertTo(ch[0], CV_8U, kYScale, 16.0);
    ch[1].convertTo(ch[1], CV_8U, kCScale, 128.0 * (1.0 - kCScale));
    ch[2].convertTo(ch[2], CV_8U, kCScale, 128.0 * (1.0 - kCScale));

    // Y plane straight in.
    ch[0].copyTo(nv12(cv::Rect(at.x, at.y, bgr.cols, bgr.rows)));
    // Chroma averaged down 2x (INTER_AREA) and interleaved as U,V = Cb,Cr.
    cv::Mat u2, v2;
    cv::resize(ch[2], u2, cv::Size(bgr.cols / 2, bgr.rows / 2), 0, 0, cv::INTER_AREA);
    cv::resize(ch[1], v2, cv::Size(bgr.cols / 2, bgr.rows / 2), 0, 0, cv::INTER_AREA);
    std::vector<cv::Mat> planes{u2, v2};
    cv::Mat uv;
    cv::merge(planes, uv);                  // CV_8UC2, UVUV interleaved
    cv::Mat uvBytes = uv.reshape(1, u2.rows); // CV_8UC1, width == bgr.cols
    uvBytes.copyTo(nv12(cv::Rect(at.x, H + at.y / 2, bgr.cols, bgr.rows / 2)));
}

bool Camera::read(cv::Mat& frame) {
    return cap_.read(frame) && !frame.empty();
}

} // namespace olc
