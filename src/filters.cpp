#include "filters.hpp"

#include <glob.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

#include "pig3d.hpp"

namespace olc {

namespace {

namespace mpv = ::mediapipe::tasks::vision;
using MpFaceLandmarker = mpv::face_landmarker::FaceLandmarker;
using MpFaceLandmarkerOptions = mpv::face_landmarker::FaceLandmarkerOptions;
using MpFaceLandmarkerResult = mpv::face_landmarker::FaceLandmarkerResult;

// Detection settings, sized for the Raspberry Pi 5 (BCM2712, Cortex-A76 at
// 2.4 GHz). Inference runs on every frame: the landmarker is in VIDEO mode, so
// MediaPipe re-runs the *detector* only when it loses tracking and otherwise
// just refreshes the mesh from the previous frame's ROI -- its own throttle,
// and a better one than skipping frames, which left the warp lagging the face.
//
// Inference still runs on a downscaled copy, but at 640 px rather than the
// 320 px a slower board needs: the mesh model crops the face out of this image
// before resizing it to its own input, so a wider detection image is what lets
// a small or distant face keep enough detail to land accurate landmarks.
constexpr double kDetectWidth = 640.0;
constexpr int kMaxFaces = 4; // per-face mesh inference is cheap enough here
// Nominal spacing between frames. VIDEO mode only needs timestamps that
// increase; the exact value just has to be plausible for its tracking.
constexpr int64_t kFrameIntervalMs = 33;
constexpr double kTearSpeed = 0.019; // tear cycle progress per frame (fall speed)

// --- MediaPipe canonical face-mesh indices --------------------------------
// The mesh is the standard 468-point topology, plus the two 5-point irises
// (468..477) when the bundle includes the attention/iris model. "Left" and
// "right" here are the *subject's*, which is how MediaPipe labels them; they
// are resolved into on-screen order in detect() so a mirrored preview works.
constexpr int kMeshWithIris = 478;
constexpr int kNoseTip = 1;
constexpr int kChin = 152;
constexpr int kForehead = 10;
constexpr int kCheekR = 234, kCheekL = 454; // face-oval extremes
constexpr int kMouthR = 61, kMouthL = 291;  // mouth corners
constexpr int kLipInnerTop = 13, kLipInnerBot = 14;
constexpr int kBrowInnerR = 107, kBrowInnerL = 336;
constexpr int kIrisR = 468, kIrisL = 473;
constexpr int kEyeROuter = 33, kEyeRInner = 133, kEyeRUp = 159, kEyeRLow = 145;
constexpr int kEyeLOuter = 263, kEyeLInner = 362, kEyeLUp = 386, kEyeLLow = 374;
// Size of the base topology. Every index above lives inside it (the highest is
// the face-oval cheek at 454), so a shorter mesh is not the topology this code
// understands and is skipped rather than read out of bounds.
constexpr int kMeshMin = 468;

// How strongly the measured landmark asymmetries map onto head rotation. Both
// are approximations of a full 3D pose fit, calibrated so a head turned or
// tipped ~30 degrees produces roughly that much rotation on the pig.
constexpr float kYawGain = 1.5f;
constexpr float kPitchGain = 3.5f;
// Where the eye line sits between the forehead and the chin on a level head.
constexpr float kNeutralEyeT = 0.42f;

// Where a packaged Face Landmarker bundle is looked for, in order. The
// media-pipe-builder .deb installs under a versioned /opt/mediapipe/<ver> and
// points /opt/mediapipe/current at the newest one.
const char* kModelGlobs[] = {
    "/opt/mediapipe/current/share/mediapipe/models/face_landmarker.task",
    "/opt/mediapipe/*/share/mediapipe/models/face_landmarker.task",
    "/usr/share/mediapipe/models/face_landmarker.task",
    "/usr/local/share/mediapipe/models/face_landmarker.task",
    "/usr/share/open-lego-camera/face_landmarker.task",
    "/usr/local/share/open-lego-camera/face_landmarker.task",
    "face_landmarker.task",
};

float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

float len(const cv::Point2f& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

float dot(const cv::Point2f& a, const cv::Point2f& b) { return a.x * b.x + a.y * b.y; }

// Alpha-blend a filled circle onto a bounded ROI of `img` (keeps the cost of
// each tear tiny, and gives the tears their translucent, watery look).
void blendCircle(cv::Mat& img, cv::Point c, int r, cv::Scalar col, double a) {
    if (r < 1) r = 1;
    cv::Rect rc(c.x - r, c.y - r, 2 * r + 1, 2 * r + 1);
    rc &= cv::Rect(0, 0, img.cols, img.rows);
    if (rc.area() <= 0) return;
    cv::Mat roi = img(rc), ov = roi.clone();
    cv::circle(ov, c - rc.tl(), r, col, cv::FILLED, cv::LINE_AA);
    cv::addWeighted(ov, a, roi, 1.0 - a, 0.0, roi);
}

// Alpha-blend a thick line (a tear trail) onto a bounded ROI of `img`.
void blendLine(cv::Mat& img, cv::Point a, cv::Point b, cv::Scalar col, int th,
               double alpha) {
    if (th < 1) th = 1;
    int minx = std::min(a.x, b.x) - th, maxx = std::max(a.x, b.x) + th;
    int miny = std::min(a.y, b.y) - th, maxy = std::max(a.y, b.y) + th;
    cv::Rect rc(minx, miny, maxx - minx + 1, maxy - miny + 1);
    rc &= cv::Rect(0, 0, img.cols, img.rows);
    if (rc.area() <= 0) return;
    cv::Mat roi = img(rc), ov = roi.clone();
    cv::line(ov, a - rc.tl(), b - rc.tl(), col, th, cv::LINE_AA);
    cv::addWeighted(ov, alpha, roi, 1.0 - alpha, 0.0, roi);
}

// Draw a single tear at cycle progress p (0..1) rolling from `origin` down the
// cheek. `down` and `side` are the head's unit down/right axes, so the tear
// runs down the face even when the head is tilted. The motion is modelled on a
// real tear rather than a raindrop: it first *wells* at the eye as a growing
// bead, then a single droplet releases and rolls down, starting slow (held by
// surface tension) and accelerating, drifting slightly outward along the cheek
// and fading as it dries near the jaw. It leaves a thin glistening wet track
// and carries a bright highlight.
void drawTear(cv::Mat& img, cv::Point2f origin, cv::Point2f down, cv::Point2f side,
              float fall, float fw, float dir, float p) {
    const cv::Scalar track(235, 210, 165); // BGR: faint bluish wet streak
    const cv::Scalar body(250, 235, 205);  // translucent watery droplet
    const cv::Scalar shine(255, 255, 255);
    const float beadR = std::max(2.f, 0.024f * fw);
    const float wellEnd = 0.22f; // fraction of the cycle spent welling up
    auto pt = [](cv::Point2f v) { return cv::Point((int)v.x, (int)v.y); };

    if (p < wellEnd) {
        // Welling: a small bead pools on the lid and swells before it drops.
        float g = p / wellEnd;
        int r = std::max(1, (int)(beadR * (0.35f + 0.55f * g)));
        cv::Point2f b = origin + down * (r * 0.4f);
        blendCircle(img, pt(b), r, body, 0.30 + 0.30 * g);
        blendCircle(img, pt(b - side * (r * 0.35f) - down * (r * 0.35f)),
                    std::max(1, r / 3), shine, 0.5 + 0.25 * g);
        return;
    }

    float u = (p - wellEnd) / (1.f - wellEnd); // 0..1 along the roll
    // Ease-in: slow release, then accelerating as the drop runs (u^2-ish).
    float ease = u * u * (1.15f - 0.15f * u);
    // Gentle outward bow following the curve of the cheek.
    cv::Point2f c = origin + down * (fall * ease) +
                    side * (dir * 0.045f * fw * std::sin(u * 1.5708f));

    // Fade in at release and out as it dries near the jaw.
    float alpha = 1.f;
    if (u < 0.12f) alpha = u / 0.12f;
    else if (u > 0.82f) alpha = std::max(0.f, (1.f - u) / 0.18f);

    // Wet track: a thin streak tracing the droplet's actual path from the eye
    // down to where it is now, faintest at the top (drying) and following the
    // same slight bow.
    const int seg = 8;
    const int th = std::max(1, (int)(beadR * 0.35f));
    cv::Point2f prev = origin;
    for (int s = 1; s <= seg; ++s) {
        float w = u * (float)s / seg; // sample the travelled path
        float we = w * w * (1.15f - 0.15f * w);
        cv::Point2f t = origin + down * (fall * we) +
                        side * (dir * 0.045f * fw * std::sin(w * 1.5708f));
        blendLine(img, pt(prev), pt(t), track, th,
                  0.22 * alpha * (float)s / seg); // fade toward the eye
        prev = t;
    }

    // The droplet: a slightly teardrop-shaped bead with a bright highlight.
    int r = std::max(2, (int)beadR);
    blendCircle(img, pt(c), r, body, 0.60 * alpha);
    blendCircle(img, pt(c - down * (r * 0.7f)), std::max(1, (int)(r * 0.6f)), body,
                0.55 * alpha); // pointed top -> teardrop silhouette
    blendCircle(img, pt(c - side * (r * 0.33f) - down * (r * 0.33f)),
                std::max(1, r / 3), shine, 0.85 * alpha);
}

// Locally reshape `img` so that the image feature at each src[i] appears to move
// to dst[i], with a smooth Gaussian falloff of radius sig[i]. Implemented as an
// inverse map for cv::remap: for an output pixel p the source sample is
//   p - sum_i w_i(p) * (dst[i] - src[i]),   w_i(p) = exp(-|p-dst[i]|^2 / 2sig^2)
// so at p == dst[i] the sample is exactly src[i]. Only the affected bounding box
// is remapped, so the work stays proportional to the reshaped region.
//
// The Gaussian is *separable* -- exp(-(ex^2+ey^2)/2s^2) is exp(-ex^2/2s^2)
// times exp(-ey^2/2s^2) -- so each control point's column factors are computed
// once for the whole region and its row factor once per row, leaving only
// multiply-accumulate in the inner loop. Evaluating exp() per pixel per control
// point instead (the obvious way) costs width*height*n transcendentals, which
// at 1080p with a face filling the frame is millions per frame; this is
// width+height per point. Each point is also only visited inside the box where
// it still has weight, rather than over the whole union region.
//
// That box is 4.5 sigma, not the 3 sigma used to pad the region below. The two
// numbers are doing different jobs: 3 sigma bounds how far the *region* has to
// extend, where being slightly small costs nothing, but truncating a point's
// influence at 3 sigma would drop a weight of exp(-4.5) ~ 1.1e-2 -- half a
// pixel of displacement, appearing as a hard seam in the warp field. At
// 4.5 sigma the dropped weight is ~4e-5, genuinely invisible.
void warpRegion(cv::Mat& img, const std::vector<cv::Point2f>& src,
                const std::vector<cv::Point2f>& dst,
                const std::vector<float>& sig) {
    const size_t n = src.size();
    if (n == 0 || dst.size() != n || sig.size() != n) return;

    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f, maxsig = 1.f;
    for (size_t i = 0; i < n; ++i) {
        minx = std::min({minx, dst[i].x, src[i].x});
        miny = std::min({miny, dst[i].y, src[i].y});
        maxx = std::max({maxx, dst[i].x, src[i].x});
        maxy = std::max({maxy, dst[i].y, src[i].y});
        maxsig = std::max(maxsig, sig[i]);
    }
    int pad = (int)std::ceil(3.f * maxsig); // Gaussian is negligible past ~3 sigma
    cv::Rect roi((int)std::floor(minx) - pad, (int)std::floor(miny) - pad,
                 (int)std::ceil(maxx - minx) + 2 * pad,
                 (int)std::ceil(maxy - miny) + 2 * pad);
    roi &= cv::Rect(0, 0, img.cols, img.rows);
    if (roi.width < 3 || roi.height < 3) return;

    // Start from the identity map; each control point then subtracts its
    // weighted displacement over the pixels it actually reaches.
    cv::Mat mapx(roi.height, roi.width, CV_32F);
    cv::Mat mapy(roi.height, roi.width, CV_32F);
    for (int yy = 0; yy < roi.height; ++yy) {
        float* mx = mapx.ptr<float>(yy);
        float* my = mapy.ptr<float>(yy);
        const float ay = (float)(roi.y + yy);
        for (int xx = 0; xx < roi.width; ++xx) {
            mx[xx] = (float)(roi.x + xx);
            my[xx] = ay;
        }
    }

    std::vector<float> gx; // column factors for the control point in hand
    for (size_t i = 0; i < n; ++i) {
        const float dX = dst[i].x - src[i].x;
        const float dY = dst[i].y - src[i].y;
        const float inv = 1.f / (2.f * sig[i] * sig[i]);
        const float reach = 4.5f * sig[i];

        // The box where this point still has weight, in roi-local coordinates.
        int x0 = std::max(0, (int)std::floor(dst[i].x - reach) - roi.x);
        int x1 = std::min(roi.width, (int)std::ceil(dst[i].x + reach) - roi.x + 1);
        int y0 = std::max(0, (int)std::floor(dst[i].y - reach) - roi.y);
        int y1 = std::min(roi.height, (int)std::ceil(dst[i].y + reach) - roi.y + 1);
        if (x0 >= x1 || y0 >= y1) continue;

        gx.resize((size_t)(x1 - x0));
        for (int xx = x0; xx < x1; ++xx) {
            const float ex = (float)(roi.x + xx) - dst[i].x;
            gx[(size_t)(xx - x0)] = std::exp(-ex * ex * inv);
        }

        for (int yy = y0; yy < y1; ++yy) {
            const float ey = (float)(roi.y + yy) - dst[i].y;
            const float gy = std::exp(-ey * ey * inv);
            float* mx = mapx.ptr<float>(yy);
            float* my = mapy.ptr<float>(yy);
            for (int xx = x0; xx < x1; ++xx) {
                const float w = gy * gx[(size_t)(xx - x0)];
                mx[xx] -= w * dX;
                my[xx] -= w * dY;
            }
        }
    }

    cv::Mat warped;
    cv::remap(img, warped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    warped.copyTo(img(roi));
}

// Score of one MediaPipe blendshape by its ARKit-style name, or -1 when the
// bundle was loaded without the blendshape head.
float blendshape(const MpFaceLandmarkerResult& r, size_t face, const char* name) {
    if (!r.face_blendshapes || face >= r.face_blendshapes->size()) return -1.f;
    for (const auto& c : (*r.face_blendshapes)[face].categories)
        if (c.category_name && *c.category_name == name) return c.score;
    return -1.f;
}

// Wrap a BGR-or-luma OpenCV image as an RGB mediapipe::Image, downscaled to the
// detection width. MediaPipe owns its own buffer, so this copies once -- at
// detection resolution, which is a fraction of the frame.
mediapipe::Image toMpImage(const cv::Mat& src) {
    double scale = kDetectWidth / std::max(1, src.cols);
    cv::Mat small;
    if (scale < 1.0)
        cv::resize(src, small, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        small = src;

    auto frame = std::make_shared<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, small.cols, small.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat dst = mediapipe::formats::MatView(frame.get());
    // Callers normally hand over colour (see Camera::nv12ToBGRScaled, which
    // produces the detection image without a full-frame convert). A plain luma
    // image is still accepted and replicated into RGB -- the landmarker copes
    // with grayscale, just less well in awkward lighting.
    cv::cvtColor(small, dst,
                 small.channels() == 1 ? cv::COLOR_GRAY2RGB : cv::COLOR_BGR2RGB);
    return mediapipe::Image(std::move(frame));
}

// Build the landmarker for `path`. Returns nullptr on failure; only the final
// attempt logs, since the blendshape probe below is expected to fail on
// bundles that do not carry that head.
std::unique_ptr<MpFaceLandmarker> createLandmarker(const std::string& path,
                                                   bool blendshapes, bool quiet) {
    auto opts = std::make_unique<MpFaceLandmarkerOptions>();
    opts->base_options.model_asset_path = path;
    opts->running_mode = mpv::core::RunningMode::VIDEO;
    opts->num_faces = kMaxFaces;
    opts->output_face_blendshapes = blendshapes;
    auto created = MpFaceLandmarker::Create(std::move(opts));
    if (!created.ok()) {
        if (!quiet)
            std::cerr << "filters: could not create the face landmarker from "
                      << path << ": " << created.status().ToString() << "\n";
        return nullptr;
    }
    return std::move(*created);
}

} // namespace

// The MediaPipe landmarker, kept out of filters.hpp so nothing else has to see
// (or compile against) the MediaPipe, Abseil and TFLite headers.
struct FaceFilter::Landmarker {
    std::unique_ptr<MpFaceLandmarker> task;
    bool blendshapes = false; // did the bundle give us a blendshape head?
    int64_t ts = 0;           // monotonically increasing VIDEO-mode timestamp

    // Create the landmarker for `p`, preferring a bundle that also predicts
    // blendshapes (they give `jawOpen` / `mouthSmile*` directly). Bundles
    // without that head make Create() fail, so fall back to a plain one and
    // measure the mouth geometrically instead.
    bool load(const std::string& p) {
        auto t = createLandmarker(p, /*blendshapes=*/true, /*quiet=*/true);
        blendshapes = (t != nullptr);
        if (!t) t = createLandmarker(p, /*blendshapes=*/false, /*quiet=*/false);
        if (!t) return false;
        task = std::move(t);
        if (!blendshapes)
            std::cerr << "filters: " << p << " has no blendshape head; mouth "
                         "openness will be measured from the lip landmarks\n";
        return true;
    }
};

std::string findFaceLandmarkerModel() {
    std::string found;
    for (const char* pattern : kModelGlobs) {
        glob_t g{};
        if (glob(pattern, 0, nullptr, &g) == 0 && g.gl_pathc > 0) {
            // glob() sorts its matches, so the last hit of a versioned
            // wildcard is the newest install.
            found = g.gl_pathv[g.gl_pathc - 1];
        }
        globfree(&g);
        if (!found.empty()) break;
    }
    return found;
}

FaceFilter::FaceFilter() : lm_(std::make_unique<Landmarker>()) {
    const std::string path = findFaceLandmarkerModel();
    if (!path.empty()) lm_->load(path);
}

FaceFilter::~FaceFilter() = default;

bool FaceFilter::ready() const { return lm_ && lm_->task != nullptr; }

void FaceFilter::setModel(const std::string& path) {
    if (path.empty()) return;
    Landmarker fresh;
    if (fresh.load(path)) {
        *lm_ = std::move(fresh);
    } else {
        std::cerr << "filters: could not load face model " << path
                  << "; keeping the previously loaded one\n";
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

void FaceFilter::detect(const cv::Mat& src, cv::Size frameSize) {
    faces_.clear();

    lm_->ts += kFrameIntervalMs;
    auto result = lm_->task->DetectForVideo(toMpImage(src), lm_->ts);
    // A transient inference failure just means no filtering this frame.
    if (!result.ok()) return;

    // Landmarks come back normalized to the *detection* image, and everything
    // downstream (dirtyRegion, applyRegion, the warps) works in frame
    // coordinates. Scale by the frame, which is not necessarily `src`: callers
    // on the NV12 path hand over an already-downscaled image.
    const float W = (float)frameSize.width, H = (float)frameSize.height;
    for (size_t i = 0; i < result->face_landmarks.size(); ++i) {
        const auto& mesh = result->face_landmarks[i].landmarks;
        if ((int)mesh.size() < kMeshMin) continue;

        // MediaPipe reports normalized coordinates, so the detection downscale
        // never enters the maths: scaling by the full frame size is enough.
        auto P = [&](int idx) {
            return cv::Point2f(mesh[idx].x * W, mesh[idx].y * H);
        };

        Face f;

        // Tight mesh bounding box, clamped to the frame.
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (const auto& p : mesh) {
            x0 = std::min(x0, p.x * W);
            x1 = std::max(x1, p.x * W);
            y0 = std::min(y0, p.y * H);
            y1 = std::max(y1, p.y * H);
        }
        // Clamp to the frame, not to `src` -- on the NV12 path `src` is the
        // downscaled detection image, and clamping to it would squeeze every
        // box into the top-left corner of the frame.
        f.box = cv::Rect(cv::Point((int)std::floor(x0), (int)std::floor(y0)),
                         cv::Point((int)std::ceil(x1), (int)std::ceil(y1))) &
                cv::Rect(0, 0, frameSize.width, frameSize.height);
        // Too small to reshape cleanly (matches the old detector's floor).
        if (f.box.width < 40 || f.box.height < 40) continue;

        // Eye centres: the iris centre when the bundle carries the iris mesh,
        // otherwise the middle of the four eye-contour extremes.
        const bool iris = (int)mesh.size() >= kMeshWithIris;
        auto eye = [&](int ir, int outer, int inner, int up, int low) {
            if (iris) return P(ir);
            return (P(outer) + P(inner) + P(up) + P(low)) * 0.25f;
        };
        const cv::Point2f eyeSubjR =
            eye(kIrisR, kEyeROuter, kEyeRInner, kEyeRUp, kEyeRLow);
        const cv::Point2f eyeSubjL =
            eye(kIrisL, kEyeLOuter, kEyeLInner, kEyeLUp, kEyeLLow);

        // MediaPipe labels landmarks by the subject's anatomy, which lands on
        // the image left or the image right depending on whether the preview is
        // mirrored. Settle that once from the eyes, then order every other pair
        // the same way, so all the geometry below is in plain screen terms.
        const bool mirrored = eyeSubjL.x < eyeSubjR.x;
        auto ordered = [&](cv::Point2f subjRight, cv::Point2f subjLeft) {
            return mirrored ? std::make_pair(subjLeft, subjRight)
                            : std::make_pair(subjRight, subjLeft);
        };
        std::tie(f.eyeL, f.eyeR) = ordered(eyeSubjR, eyeSubjL);
        std::tie(f.mouthL, f.mouthR) = ordered(P(kMouthR), P(kMouthL));
        std::tie(f.browL, f.browR) = ordered(P(kBrowInnerR), P(kBrowInnerL));
        std::tie(f.lidL, f.lidR) = ordered(P(kEyeRLow), P(kEyeLLow));
        const auto [cheekL, cheekR] = ordered(P(kCheekR), P(kCheekL));
        f.lipTop = P(kLipInnerTop);
        f.lipBot = P(kLipInnerBot);

        // Head axes from the eye line: `right` runs ear-to-ear, `down` toward
        // the chin. Everything below is expressed in those, so the filters
        // follow a tilted head instead of the image axes.
        const cv::Point2f eyeVec = f.eyeR - f.eyeL;
        const float eyeSep = len(eyeVec);
        if (eyeSep < 8.f) continue; // degenerate: nothing to anchor to
        f.right = eyeVec * (1.f / eyeSep);
        f.down = cv::Point2f(-f.right.y, f.right.x);

        // Yaw: turning the head slides the nose toward one cheek, so how the
        // nose divides the ear-to-ear span is a direct read of the turn.
        const cv::Point2f nose = P(kNoseTip);
        const float dL = dot(nose - cheekL, f.right);
        const float dR = dot(cheekR - nose, f.right);
        if (dL + dR > 1e-3f)
            f.yaw = clampf(kYawGain * (dL - dR) / (dL + dR), -0.95f, 0.95f);

        // Pitch: nodding foreshortens the forehead or the jaw, sliding the eye
        // line along the forehead-to-chin span.
        const cv::Point2f fore = P(kForehead), chin = P(kChin);
        const float faceLen = dot(chin - fore, f.down);
        if (faceLen > 1e-3f) {
            const float t = dot((f.eyeL + f.eyeR) * 0.5f - fore, f.down) / faceLen;
            f.pitch = clampf(kPitchGain * (kNeutralEyeT - t), -0.6f, 0.6f);
        }

        // Expression. The blendshape head reads the mouth straight off the
        // mesh; without it, fall back to the inner-lip gap.
        float jaw = blendshape(*result, i, "jawOpen");
        if (jaw < 0.f) {
            const float mouthW = len(f.mouthR - f.mouthL);
            jaw = mouthW > 1e-3f ? len(f.lipBot - f.lipTop) / (0.45f * mouthW) : 0.f;
        }
        f.open = clamp01(jaw);
        const float sl = blendshape(*result, i, "mouthSmileLeft");
        const float sr = blendshape(*result, i, "mouthSmileRight");
        f.smile = (sl >= 0.f && sr >= 0.f) ? clamp01(0.5f * (sl + sr)) : 0.f;

        faces_.push_back(std::move(f));
    }
}

void FaceFilter::apply(cv::Mat& frame, Filter filter, double phase) {
    if (filter == Filter::None || frame.empty()) return;
    if (frame.type() != CV_8UC3) return; // filters assume BGR 8-bit
    updateDetection(frame);
    applyRegion(frame, {0, 0}, filter, phase);
}

int FaceFilter::detectionWidth() { return (int)kDetectWidth; }

void FaceFilter::updateDetection(const cv::Mat& src, cv::Size frameSize) {
    if (!ready()) {
        if (!warned_) {
            std::cerr << "filters: no MediaPipe face model loaded; facial "
                         "filters disabled. Install the MediaPipe package or "
                         "pass --face-model PATH.\n";
            warned_ = true;
        }
        return;
    }
    if (src.empty()) {
        // The model is loaded but the caller handed over nothing, so the
        // filters would quietly do nothing on every frame. Say so once rather
        // than looking like the effect is simply broken.
        if (!warned_) {
            std::cerr << "filters: empty detection image; facial filters have "
                         "nothing to run on (the camera frame could not be "
                         "converted).\n";
            warned_ = true;
        }
        return;
    }
    // An unset frameSize means `src` is the full-resolution frame.
    detect(src, frameSize.area() > 0 ? frameSize : src.size());
}

cv::Rect FaceFilter::dirtyRegion(Filter filter, int w, int h) const {
    if (filter == Filter::None || w <= 0 || h <= 0) return cv::Rect();

    // Union the per-face bounding boxes, each grown to cover the warp's
    // Gaussian falloff (~half a face width) and the tears that fall down the
    // cheeks below the eyes. One face -> a tight box; several -> a larger box,
    // still far cheaper than converting the whole frame.
    //
    // The pig-face ears rise well above the head and the snout/cheeks spread to
    // the sides, so that filter needs a noticeably larger margin than the warps.
    const bool pig = (filter == Filter::PigFace);
    cv::Rect uni;
    for (const Face& face : faces_) {
        const cv::Rect& f = face.box;
        int mx = pig ? std::max(10, f.width * 9 / 10) : std::max(8, f.width * 2 / 5);
        int mtop = pig ? std::max(10, f.height) : std::max(6, f.height * 3 / 10);
        int mbot = pig ? std::max(10, f.height * 2 / 5) : std::max(8, f.height / 2);
        cv::Rect r(f.x - mx, f.y - mtop, f.width + 2 * mx, f.height + mtop + mbot);
        uni = (uni.area() == 0) ? r : (uni | r);
    }
    uni &= cv::Rect(0, 0, w, h);
    if (uni.area() == 0) return cv::Rect();

    // Snap to an even grid so the crop lines up with NV12's 2x2 chroma plane.
    int x0 = uni.x & ~1, y0 = uni.y & ~1;
    int x1 = (uni.x + uni.width) & ~1, y1 = (uni.y + uni.height) & ~1;
    if (x1 - x0 < 4 || y1 - y0 < 4) return cv::Rect();
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

void FaceFilter::applyRegion(cv::Mat& roi, cv::Point origin, Filter filter,
                             double phase) {
    if (filter == Filter::None || roi.empty() || roi.type() != CV_8UC3) return;
    // Landmarks are in frame space; shifting by the region origin puts them in
    // roi-local coordinates. The reshaping helpers clip to roi's bounds, so a
    // face only partly inside the region is handled safely.
    const cv::Point2f off((float)origin.x, (float)origin.y);
    for (const Face& f : faces_) {
        if (filter == Filter::BigSmile) {
            applySmile(roi, f, off);
        } else if (filter == Filter::Crying) {
            applyCry(roi, f, off, phase);
        } else if (filter == Filter::PigFace) {
            applyPig(roi, f, off, phase);
        }
    }
}

// ---------------------------------------------------------------------------
// The filters themselves
// ---------------------------------------------------------------------------

void FaceFilter::whitenTeeth(cv::Mat& frame, cv::Point2f cornerL,
                             cv::Point2f cornerR, cv::Point2f lipTop,
                             cv::Point2f lipBot, float strength) const {
    const std::vector<cv::Point2f> quad{cornerL, cornerR, lipTop, lipBot};
    cv::Rect m = cv::boundingRect(quad);
    m &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (m.area() < 20) return;

    cv::Mat roi = frame(m);
    for (int y = 0; y < roi.rows; ++y) {
        cv::Vec3b* row = roi.ptr<cv::Vec3b>(y);
        for (int x = 0; x < roi.cols; ++x) {
            cv::Vec3b& px = row[x];
            // Rec.601 luma; only already-bright pixels (the teeth) get whitened.
            float luma = 0.114f * px[0] + 0.587f * px[1] + 0.299f * px[2];
            if (luma <= 135.f) continue;
            float t = strength * clamp01((luma - 135.f) / 110.f);
            for (int c = 0; c < 3; ++c)
                px[c] = cv::saturate_cast<uchar>(px[c] + t * (255.f - px[c]));
        }
    }
}

void FaceFilter::applySmile(cv::Mat& frame, const Face& f, cv::Point2f off) const {
    const cv::Point2f cl = f.mouthL - off, cr = f.mouthR - off;
    const cv::Point2f top = f.lipTop - off, bot = f.lipBot - off;
    const float mw = len(cr - cl);
    if (mw < 12.f) return;

    // A mouth that is already grinning needs less help; stacking a full-strength
    // grin on top of a real one is what makes this kind of filter look rubbery.
    const float damp = 1.f - 0.35f * f.smile;
    const float outX = 0.28f * mw * damp;                // corners pull outward
    const float upY = 0.17f * mw * damp;                 // ...and upward
    const float openY = (0.055f + 0.165f * f.open) * mw; // vertical stretch

    std::vector<cv::Point2f> src, dst;
    std::vector<float> sig;
    // Mouth corners -> up and out along the head's own axes (the grin).
    const cv::Point2f dstL = cl - f.right * outX - f.down * upY;
    const cv::Point2f dstR = cr + f.right * outX - f.down * upY;
    src.push_back(cl); dst.push_back(dstL); sig.push_back(0.44f * mw);
    src.push_back(cr); dst.push_back(dstR); sig.push_back(0.44f * mw);
    // Upper lip up / lower lip down -> open the mouth so the teeth show.
    const cv::Point2f dstTop = top - f.down * openY;
    const cv::Point2f dstBot = bot + f.down * openY;
    src.push_back(top); dst.push_back(dstTop); sig.push_back(0.36f * mw);
    src.push_back(bot); dst.push_back(dstBot); sig.push_back(0.36f * mw);

    warpRegion(frame, src, dst, sig);
    // Whiten what the warp just exposed, so the teeth band follows the grin.
    // Teeth only show in the middle of the mouth, so pull the horizontal span
    // in from the corners -- otherwise a bright cheek next to a wide grin gets
    // brightened too.
    const cv::Point2f mid = (dstL + dstR) * 0.5f;
    const cv::Point2f half = (dstR - dstL) * 0.35f;
    whitenTeeth(frame, mid - half, mid + half, dstTop, dstBot,
                0.30f + 0.50f * f.open);
}

void FaceFilter::drawTears(cv::Mat& frame, const Face& f, cv::Point2f off,
                           double phase) const {
    const float eyeSep = len(f.eyeR - f.eyeL);
    // The tear geometry used to be scaled off the detector's face box. The mesh
    // has no such box, so it is scaled off the eye separation instead (a face is
    // roughly 2.4 eye-separations wide).
    const float fw = 2.4f * eyeSep;
    // Two tear tracks under each eye (inner + outer corner), each shedding a
    // little stream of droplets. Everything is phase-staggered so it reads as
    // heavy weeping -- lots of tears -- rather than a curtain of rain.
    struct Track { cv::Point2f origin; float dir, fall, phase; };
    const cv::Point2f lidL = f.lidL - off, lidR = f.lidR - off;
    const cv::Point2f step = f.right * (0.12f * eyeSep);
    const Track tracks[] = {
        {lidL - step, -1.00f, 1.00f * eyeSep, 0.00f}, // left eye, outer corner
        {lidL + step, -0.35f, 0.90f * eyeSep, 0.29f}, // left eye, inner corner
        {lidR - step, +0.35f, 0.90f * eyeSep, 0.61f}, // right eye, inner corner
        {lidR + step, +1.00f, 1.00f * eyeSep, 0.83f}, // right eye, outer corner
    };
    // Several droplets per track, spread across the cycle so a tear is welling,
    // rolling and drying on each cheek at once.
    const float dropOff[] = {0.0f, 0.34f, 0.67f};
    for (const Track& t : tracks) {
        for (float d : dropOff) {
            float p = (float)std::fmod(phase * kTearSpeed + t.phase + d, 1.0);
            drawTear(frame, t.origin, f.down, f.right, t.fall, fw, t.dir, p);
        }
    }
}

void FaceFilter::applyCry(cv::Mat& frame, const Face& f, cv::Point2f off,
                          double phase) const {
    const cv::Point2f cl = f.mouthL - off, cr = f.mouthR - off;
    const cv::Point2f top = f.lipTop - off;
    const cv::Point2f bl = f.browL - off, br = f.browR - off;
    const float mw = len(cr - cl);
    if (mw < 12.f) return;

    const float downY = 0.17f * mw; // corners sink
    const float inX = 0.083f * mw;  // ...and draw slightly inward

    std::vector<cv::Point2f> src, dst;
    std::vector<float> sig;
    // Mouth corners down + centre up -> a sad frown (inverse of the smile).
    src.push_back(cl);
    dst.push_back(cl + f.right * inX + f.down * downY);
    sig.push_back(0.42f * mw);
    src.push_back(cr);
    dst.push_back(cr - f.right * inX + f.down * downY);
    sig.push_back(0.42f * mw);
    src.push_back(top);
    dst.push_back(top - f.down * (0.11f * mw));
    sig.push_back(0.36f * mw);
    // Inner brows down and together -> the pinched, crumpled crying brow.
    src.push_back(bl);
    dst.push_back(bl + f.right * inX + f.down * (0.17f * mw));
    sig.push_back(0.33f * mw);
    src.push_back(br);
    dst.push_back(br - f.right * inX + f.down * (0.17f * mw));
    sig.push_back(0.33f * mw);

    warpRegion(frame, src, dst, sig);
    drawTears(frame, f, off, phase);
}

void FaceFilter::applyPig(cv::Mat& frame, const Face& f, cv::Point2f off,
                          double phase) const {
    // The pig is a set of real 3D meshes (ears + snout) rendered through a
    // perspective camera by pig3d. Keeping the graphics three-dimensional is
    // what makes them share the face's orientation and perspective -- the snout
    // protrudes and foreshortens, the ears swing around and occlude behind the
    // head as it turns -- instead of looking like flat stickers. The face mesh
    // supplies the pose, so the pig follows a turned or tipped head properly
    // rather than guessing from a pair of eye boxes inside a detector box.
    pig3d::Head head;
    head.hasEyes = true;
    head.leftEye = f.eyeL - off;
    head.rightEye = f.eyeR - off;
    head.hasPose = true;
    head.yaw = f.yaw;
    head.pitch = f.pitch;
    const cv::Rect box(f.box.x - (int)off.x, f.box.y - (int)off.y, f.box.width,
                       f.box.height);
    pig3d::render(frame, box, head, phase);
}

void FaceFilter::drawPigPreview(cv::Mat& frame, const cv::Rect& face,
                                cv::Point2f leftEye, cv::Point2f rightEye,
                                double phase) const {
    if (frame.empty() || frame.type() != CV_8UC3) return;
    const bool hasEyes = (leftEye.x >= 0.f && rightEye.x >= 0.f);
    pig3d::render(frame, face, hasEyes, leftEye, rightEye, phase);
}

Filter nextFilter(Filter f) {
    switch (f) {
        case Filter::None:     return Filter::BigSmile;
        case Filter::BigSmile: return Filter::Crying;
        case Filter::Crying:   return Filter::PigFace;
        case Filter::PigFace:  return Filter::None;
    }
    return Filter::None;
}

const char* filterName(Filter f) {
    switch (f) {
        case Filter::None:     return "Filter Off";
        case Filter::BigSmile: return "Big Smile";
        case Filter::Crying:   return "Crying";
        case Filter::PigFace:  return "Pig Face";
    }
    return "";
}

} // namespace olc
