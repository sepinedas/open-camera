#include "filters.hpp"

#include "face3d.hpp"

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
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_connections.h"


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

// Face-mesh overlay styling. Drawn directly onto the frame, so the colours
// below are what lands on screen.
const cv::Scalar kMeshEdge(170, 235, 110);    // BGR: cool green, reads on skin
const cv::Scalar kMeshFeature(120, 245, 245); // contours, picked out brighter
const cv::Scalar kMeshDot(245, 255, 245);

// The mesh's connectivity comes from MediaPipe itself -- the same tables its
// own renderers use -- rather than being re-derived here. kFaceLandmarksTesselation
// is the full 2556-edge triangle net; the feature tables are the contours
// (lips, eyes, brows, irises, face oval) drawn over it in a second colour.
using MpConn = mpv::face_landmarker::FaceLandmarksConnections;

// MediaPipe stores the tessellation as edges, but they come in consecutive
// triples that close into a triangle -- {a,b},{b,c},{c,a} -- so the triangle
// list is derivable rather than a second table to carry. Verified at compile
// time so a future table reshuffle cannot silently produce garbage geometry.
constexpr int kMeshTriangles =
    (int)MpConn::kFaceLandmarksTesselation.size() / 3;
constexpr bool tesselationIsTriples() {
    for (size_t k = 0; k + 2 < MpConn::kFaceLandmarksTesselation.size(); k += 3) {
        const auto& a = MpConn::kFaceLandmarksTesselation[k];
        const auto& b = MpConn::kFaceLandmarksTesselation[k + 1];
        const auto& c = MpConn::kFaceLandmarksTesselation[k + 2];
        if (a[1] != b[0] || b[1] != c[0] || c[1] != a[0]) return false;
    }
    return true;
}
static_assert(tesselationIsTriples(),
              "MediaPipe's tessellation is no longer consecutive edge triples; "
              "the animal filters' triangle list must be rebuilt");

float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

float len(const cv::Point2f& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

float dot(const cv::Point2f& a, const cv::Point2f& b) { return a.x * b.x + a.y * b.y; }

// --- Dog markings, in the head's own frame ---------------------------------
// Coordinates are eye-separation units from the eye midpoint: +u toward the
// image-right eye, +v toward the chin. Working here rather than in pixels is
// what makes the markings track the face -- they are placed relative to the
// eyes and nose, so they hold through scale, roll, turn and expression.
const cv::Vec3f kDogBase(74, 132, 190);    // BGR: tan coat
const cv::Vec3f kDogMask(38, 64, 104);     // darker patches around the eyes
const cv::Vec3f kDogMuzzle(226, 238, 248); // pale muzzle and brow blaze
const cv::Vec3f kDogNose(26, 24, 24);      // near-black nose leather
// How far the subject's own shading may push the paint. Wide enough to keep
// real modelling, tight enough that a blown highlight or deep shadow does not
// turn the coat white or black.
constexpr float kShadeMin = 0.62f, kShadeMax = 1.42f;

// 1 inside the ellipse, easing to 0 across the outer `soft` fraction of it.
float ellipseMask(float u, float v, float cu, float cv_, float ru, float rv,
                  float soft) {
    const float du = (u - cu) / ru, dv = (v - cv_) / rv;
    const float d = std::sqrt(du * du + dv * dv);
    return clamp01((1.f - d) / std::max(0.05f, soft));
}

// As above, but with the ellipse rotated by `ang` (radians, +u toward +v).
// The scowl needs it: a brow that slopes down toward the nose cannot be drawn
// with an axis-aligned ellipse.
float ellipseMaskRot(float u, float v, float cu, float cv_, float ru, float rv,
                     float soft, float ang) {
    const float ca = std::cos(ang), sa = std::sin(ang);
    const float u0 = u - cu, v0 = v - cv_;
    const float du = (u0 * ca + v0 * sa) / ru;
    const float dv = (-u0 * sa + v0 * ca) / rv;
    const float d = std::sqrt(du * du + dv * dv);
    return clamp01((1.f - d) / std::max(0.05f, soft));
}

// Integer hash -> 0..1. No transcendentals: this runs per pixel, and a
// sin-based hash would cost more than the rest of the shading put together.
float hash21(int x, int y) {
    // Multiply as unsigned: the signed form overflows for x >= 6, which is
    // undefined behaviour, and at -O2 it does not merely give odd numbers.
    uint32_t h = (uint32_t)x * 374761393u ^ (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0xFFFFFFu) * (1.f / (float)0xFFFFFF);
}

// Smoothly interpolated value noise over that hash.
float valueNoise(float x, float y) {
    const float fx = std::floor(x), fy = std::floor(y);
    const int ix = (int)fx, iy = (int)fy;
    float sx = x - fx, sy = y - fy;
    sx = sx * sx * (3.f - 2.f * sx); // smoothstep, so cells do not show
    sy = sy * sy * (3.f - 2.f * sy);
    const float a = hash21(ix, iy), b = hash21(ix + 1, iy);
    const float c = hash21(ix, iy + 1), d = hash21(ix + 1, iy + 1);
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
}

// Fine fur detail, in the head's frame so its scale follows the face. Sampled
// much finer across the face than down it, which stretches the noise into
// strokes that read as fur. An earlier version multiplied two crossed sine
// bands; that is cheaper still, but the product is a regular lattice and came
// out looking like woven fabric. Two octaves, returning roughly -1..1.
float furAt(float u, float v) {
    const float n = valueNoise(u * 88.f, v * 20.f) +
                    0.45f * valueNoise(u * 171.f, v * 43.f);
    return (n / 1.45f) * 2.f - 1.f;
}

// A pig is pink skin rather than fur, so its markings are gentler: a deeper
// pink in the eye sockets and along the jaw, a paler patch where the snout
// sits. The bristle texture is much finer than a dog's coat too.
const cv::Vec3f kPigBase(168, 158, 236);  // BGR: pig pink
const cv::Vec3f kPigShade(140, 126, 208); // deeper pink, eye sockets and jaw
const cv::Vec3f kPigSnout(198, 184, 247); // paler patch under the 3D snout
// Green, and shaggier than either animal: the fur term is turned well up.
const cv::Vec3f kGrBase(72, 196, 96);    // BGR: vivid green coat
const cv::Vec3f kGrLight(120, 226, 150); // paler muzzle, cheeks and chin
const cv::Vec3f kGrBrow(30, 84, 42);     // the heavy scowling brow itself
const cv::Vec3f kGrDark(50, 122, 66);    // softer shading: sockets, hollows
const cv::Vec3f kGrEye(105, 228, 218);   // yellow-green around the eyes

constexpr float kDogFur = 0.085f;
constexpr float kPigFur = 0.035f;
constexpr float kGrinchFur = 0.115f;

// The coat colour at one point on the face, blended front to back.
cv::Vec3f dogColourAt(float u, float v) {
    cv::Vec3f c = kDogBase;
    auto over = [&](const cv::Vec3f& col, float a) {
        c = c * (1.f - a) + col * a;
    };
    // Dark patches around both eyes.
    over(kDogMask, ellipseMask(u, v, -0.54f, 0.00f, 0.60f, 0.48f, 0.42f));
    over(kDogMask, ellipseMask(u, v, 0.54f, 0.00f, 0.60f, 0.48f, 0.42f));
    // Pale blaze up the forehead, between the patches.
    over(kDogMuzzle, ellipseMask(u, v, 0.f, -0.62f, 0.24f, 0.82f, 0.45f));
    // Pale muzzle over the nose and mouth.
    over(kDogMuzzle, ellipseMask(u, v, 0.f, 0.98f, 0.70f, 0.74f, 0.26f));
    // Nose leather.
    over(kDogNose, ellipseMask(u, v, 0.f, 0.62f, 0.32f, 0.25f, 0.16f));
    return c;
}

cv::Vec3f grinchColourAt(float u, float v) {
    cv::Vec3f c = kGrBase;
    auto over = [&](const cv::Vec3f& col, float a) {
        c = c * (1.f - a) + col * a;
    };
    // Paler muzzle, cheeks and chin, with a brighter pair of cheek pads so
    // the lower face is modelled rather than one flat wash of green.
    over(kGrLight, ellipseMask(u, v, 0.f, 0.95f, 0.74f, 0.72f, 0.55f));
    over(kGrLight, ellipseMask(u, v, -0.72f, 0.52f, 0.40f, 0.34f, 0.9f) * 0.7f);
    over(kGrLight, ellipseMask(u, v, 0.72f, 0.52f, 0.40f, 0.34f, 0.9f) * 0.7f);
    // Yellow-green around the eyes. The mesh leaves the eye openings as holes,
    // so this rings the lids rather than covering anyone's eyes.
    over(kGrEye, ellipseMask(u, v, -0.52f, 0.00f, 0.48f, 0.34f, 0.70f));
    over(kGrEye, ellipseMask(u, v, 0.52f, 0.00f, 0.48f, 0.34f, 0.70f));
    // The scowl: a heavy brow over each eye, sloping down toward the nose.
    // Mirrored, so both inner ends drop.
    over(kGrBrow, ellipseMaskRot(u, v, -0.54f, -0.40f, 0.52f, 0.19f, 0.34f, 0.34f));
    over(kGrBrow, ellipseMaskRot(u, v, 0.54f, -0.40f, 0.52f, 0.19f, 0.34f, -0.34f));
    // A hollow just under each brow, so the eyes sit back in the skull instead
    // of being two flat yellow patches painted on the front of it. Kept narrow
    // and high: any lower and it swallows the eye colour.
    over(kGrDark, ellipseMask(u, v, -0.52f, -0.24f, 0.44f, 0.13f, 0.95f) * 0.40f);
    over(kGrDark, ellipseMask(u, v, 0.52f, -0.24f, 0.44f, 0.13f, 0.95f) * 0.40f);
    return c;
}

cv::Vec3f pigColourAt(float u, float v) {
    cv::Vec3f c = kPigBase;
    auto over = [&](const cv::Vec3f& col, float a) {
        c = c * (1.f - a) + col * a;
    };
    // Soft shading in the eye sockets, as on a real snouted head.
    over(kPigShade, ellipseMask(u, v, -0.52f, 0.02f, 0.54f, 0.42f, 0.70f));
    over(kPigShade, ellipseMask(u, v, 0.52f, 0.02f, 0.54f, 0.42f, 0.70f));
    // Pale patch the 3D snout stands on, so its base does not sit on a hard
    // colour edge when the head turns and the snout swings across it.
    over(kPigSnout, ellipseMask(u, v, 0.f, 0.80f, 0.70f, 0.66f, 0.45f));
    return c;
}

// --- MediaPipe canonical face-mesh indices --------------------------------
// The mesh is the standard 468-point topology, plus the two 5-point irises
// (468..477) when the bundle includes the attention/iris model. "Left" and
// "right" here are the *subject's*, which is how MediaPipe labels them; they
// are resolved into on-screen order in detect() so a mirrored preview works.
constexpr int kMeshWithIris = 478;
constexpr int kMouthR = 61, kMouthL = 291;  // mouth corners
constexpr int kLipInnerTop = 13, kLipInnerBot = 14;
constexpr int kBrowInnerR = 107, kBrowInnerL = 336;
constexpr int kIrisR = 468, kIrisL = 473;
constexpr int kNoseTip = 1;                   // where the 3D nose sits
constexpr int kChin = 152, kForehead = 10;    // the head's vertical axis
constexpr int kTempleR = 127, kTempleL = 356; // head width -> ear attachment
constexpr int kEyeROuter = 33, kEyeRInner = 133, kEyeRUp = 159, kEyeRLow = 145;
constexpr int kEyeLOuter = 263, kEyeLInner = 362, kEyeLUp = 386, kEyeLLow = 374;
// Size of the base topology. Every index above lives inside it (the highest is
// the face-oval cheek at 454), so a shorter mesh is not the topology this code
// understands and is skipped rather than read out of bounds.
constexpr int kMeshMin = 468;

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

// Fill one triangle with its three vertex colours interpolated across it.
// Flat-filling each triangle instead (cv::fillConvexPoly with one colour) is
// simpler but shows every one of the mesh's 852 facets, worst exactly where a
// marking has a hard edge -- the nose came out as a polygonal star.
// Fill one triangle with its three vertex colours interpolated across it, then
// add per-pixel polish: fine fur, and the *face's own* shading.
//
// The markings are low-frequency, so interpolating them per vertex is plenty.
// Fur is not -- it has to be evaluated per pixel, which is why the head-frame
// coordinates are interpolated alongside the colour.
//
// `invMeanLuma` normalises the face's brightness. Multiplying the paint by how
// light or dark the skin underneath is keeps the subject's own modelling --
// the shadow under the nose, the line of the lips, the fall-off at the jaw --
// so the dog looks painted onto a face instead of pasted over one. Normalising
// by the region's mean rather than a constant keeps that working in a dim room
// as well as a bright one.
void fillTriangleSmooth(cv::Mat& img, const cv::Point2f p[3],
                        const cv::Vec3f c[3], const cv::Point2f uv[3],
                        float invMeanLuma, float furDepth, float furCrown) {
    const float area = (p[1].x - p[0].x) * (p[2].y - p[0].y) -
                       (p[2].x - p[0].x) * (p[1].y - p[0].y);
    if (std::fabs(area) < 1e-6f) return;
    const float inv = 1.f / area; // signed, so either winding works

    int x0 = (int)std::floor(std::min({p[0].x, p[1].x, p[2].x}));
    int x1 = (int)std::ceil(std::max({p[0].x, p[1].x, p[2].x}));
    int y0 = (int)std::floor(std::min({p[0].y, p[1].y, p[2].y}));
    int y1 = (int)std::ceil(std::max({p[0].y, p[1].y, p[2].y}));
    x0 = std::max(0, x0); y0 = std::max(0, y0);
    x1 = std::min(img.cols - 1, x1); y1 = std::min(img.rows - 1, y1);

    for (int y = y0; y <= y1; ++y) {
        cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
        for (int x = x0; x <= x1; ++x) {
            const float px = x + 0.5f, py = y + 0.5f;
            const float w0 = ((p[1].x - px) * (p[2].y - py) -
                              (p[2].x - px) * (p[1].y - py)) * inv;
            const float w1 = ((p[2].x - px) * (p[0].y - py) -
                              (p[0].x - px) * (p[2].y - py)) * inv;
            const float w2 = 1.f - w0 - w1;
            // A small negative tolerance keeps shared edges from falling
            // between two triangles and leaving a seam of bare skin.
            if (w0 < -1e-3f || w1 < -1e-3f || w2 < -1e-3f) continue;
            cv::Vec3f col = c[0] * w0 + c[1] * w1 + c[2] * w2;

            // Fur, in head-frame coordinates so it tracks the face.
            const float fu = uv[0].x * w0 + uv[1].x * w1 + uv[2].x * w2;
            const float fv = uv[0].y * w0 + uv[1].y * w1 + uv[2].y * w2;
            // Shagginess can grow toward the crown: v is negative above the
            // eye line, so this leaves the muzzle smooth and roughs up the
            // forehead, which is most of what makes a coat look unkempt.
            const float depth = furDepth * (1.f + furCrown * clampf(-fv, 0.f, 1.f));
            const float fur = 1.f + depth * furAt(fu, fv);

            // The skin underneath, as a shading term. Rec.601 luma of what is
            // already in the buffer -- free, because the paint is opaque and
            // this pixel is about to be overwritten anyway.
            const cv::Vec3b& d = row[x];
            const float luma = 0.114f * d[0] + 0.587f * d[1] + 0.299f * d[2];
            const float shade = clampf(luma * invMeanLuma, kShadeMin, kShadeMax);

            const float g = fur * shade;
            for (int k = 0; k < 3; ++k)
                row[x][k] = cv::saturate_cast<uchar>(col[k] * g);
        }
    }
}

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
        f.lipTop = P(kLipInnerTop);
        f.lipBot = P(kLipInnerBot);

        // The mesh filters draw all of them, so keep the whole set. The 3D
        // copy scales z by the frame *width*, matching how MediaPipe defines
        // landmark depth ("roughly the same scale as x"), so the three axes
        // are commensurable and a basis built from them is meaningful.
        f.mesh.reserve(mesh.size());
        f.mesh3.reserve(mesh.size());
        for (int k = 0; k < (int)mesh.size(); ++k) {
            f.mesh.push_back(P(k));
            f.mesh3.emplace_back(mesh[k].x * W, mesh[k].y * H, mesh[k].z * W);
        }

        // Head axes from the eye line: `right` runs ear-to-ear, `down` toward
        // the chin. Everything below is expressed in those, so the filters
        // follow a tilted head instead of the image axes.
        const cv::Point2f eyeVec = f.eyeR - f.eyeL;
        const float eyeSep = len(eyeVec);
        if (eyeSep < 8.f) continue; // degenerate: nothing to anchor to
        f.right = eyeVec * (1.f / eyeSep);
        f.down = cv::Point2f(-f.right.y, f.right.x);

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
    cv::Rect uni;
    for (const Face& face : faces_) {
        const cv::Rect& f = face.box;
        // Three margins, because the filters reach different distances.
        //
        //  * the wireframe sits on the landmarks themselves: a couple of
        //    pixels for the dot radius and line width is enough;
        //  * the animal filters paint the mesh but *also* hang 3D ears and a
        //    muzzle well outside the face box. Measured against the canonical
        //    head: the grinch's ears reach 27% of a face-width past each side,
        //    and the pig's stand 30% of a face-height above the crown -- so the
        //    margins below are the worst case plus slack for head rotation.
        //    Too small a margin here does not merely waste the saving, it
        //    slices the ears off against the edge of the re-encoded region --
        //    which is what a 3 px margin, left over from when these filters
        //    only painted the mesh, was doing;
        //  * the warps need room for their Gaussian falloff and the tears.
        const bool wireOnly = (filter == Filter::FaceMesh);
        const bool animal = (filter == Filter::DogFace ||
                             filter == Filter::PigFace ||
                             filter == Filter::Grinch);
        int mx, mtop, mbot;
        if (wireOnly) {
            mx = mtop = mbot = 3;
        } else if (animal) {
            mx = std::max(10, f.width * 2 / 5);
            mtop = std::max(10, f.height * 2 / 5);
            mbot = std::max(8, f.height / 6);
        } else {
            mx = std::max(8, f.width * 2 / 5);
            mtop = std::max(6, f.height * 3 / 10);
            mbot = std::max(8, f.height / 2);
        }
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
        } else if (filter == Filter::FaceMesh) {
            applyFaceMesh(roi, f, off);
        } else if (filter == Filter::DogFace) {
            applyAnimalFace(roi, f, off, phase, face3d::Species::Dog);
        } else if (filter == Filter::PigFace) {
            applyAnimalFace(roi, f, off, phase, face3d::Species::Pig);
        } else if (filter == Filter::Grinch) {
            applyAnimalFace(roi, f, off, phase, face3d::Species::Grinch);
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

void FaceFilter::applyFaceMesh(cv::Mat& frame, const Face& f,
                               cv::Point2f off) const {
    // The tessellation indexes the 468-point topology; the iris rings need the
    // full 478. Anything shorter is not a mesh this can draw.
    if ((int)f.mesh.size() < 468) return;

    cv::Rect b(f.box.x - (int)off.x - 2, f.box.y - (int)off.y - 2,
               f.box.width + 4, f.box.height + 4);
    b &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (b.width < 8 || b.height < 8) return;

    // Drawn straight onto the frame. An earlier version accumulated into a
    // scratch copy of the region and blended it back once, to get a translucent
    // wireframe; that cost a full copy of the region every frame, per face, and
    // at 80% opacity bought almost nothing.
    cv::Mat ov = frame(b);
    const cv::Point org = b.tl();
    const cv::Rect local(0, 0, b.width, b.height);
    const int n = (int)f.mesh.size();

    auto at = [&](int idx) {
        const cv::Point2f q = f.mesh[idx] - off;
        return cv::Point((int)std::lround(q.x) - org.x,
                         (int)std::lround(q.y) - org.y);
    };
    auto drawEdges = [&](const auto& table, const cv::Scalar& col) {
        for (const auto& e : table) {
            if (e[0] >= n || e[1] >= n) continue; // shorter mesh than the table
            const cv::Point a = at(e[0]), c = at(e[1]);
            // A face at the edge of the crop has landmarks outside it.
            if (!local.contains(a) || !local.contains(c)) continue;
            cv::line(ov, a, c, col, 1, cv::LINE_AA);
        }
    };

    drawEdges(MpConn::kFaceLandmarksTesselation, kMeshEdge);
    // Contours over the top, so the features stay legible through the net.
    drawEdges(MpConn::kFaceLandmarksFaceOval, kMeshFeature);
    drawEdges(MpConn::kFaceLandmarksLips, kMeshFeature);
    drawEdges(MpConn::kFaceLandmarksLeftEye, kMeshFeature);
    drawEdges(MpConn::kFaceLandmarksRightEye, kMeshFeature);
    drawEdges(MpConn::kFaceLandmarksLeftEyeBrow, kMeshFeature);
    drawEdges(MpConn::kFaceLandmarksRightEyeBrow, kMeshFeature);
    drawEdges(MpConn::kFaceLandmarksLeftIris, kMeshFeature);
    drawEdges(MpConn::kFaceLandmarksRightIris, kMeshFeature);

    for (int i = 0; i < n; ++i) {
        const cv::Point d = at(i);
        if (!local.contains(d)) continue;
        cv::circle(ov, d, 1, kMeshDot, cv::FILLED, cv::LINE_AA);
    }
}

// Read the head's orientation straight off the mesh. Three landmarks give the
// axes: the outer eye corners span the head's width, the forehead-to-chin line
// its height, and their cross product the direction it faces. Because
// MediaPipe supplies a depth per landmark, this is a genuine 3D frame -- no
// guessing yaw from how the nose divides the face, and no foreshortening
// correction, which is what the old rig needed and never got quite right.
void FaceFilter::drawAnimalParts(cv::Mat& frame, const Face& f,
                                 cv::Point2f off, double phase,
                                 face3d::Species species) const {
    if ((int)f.mesh3.size() < 468) return;
    auto V = [&](int i) {
        return cv::Vec3f(f.mesh3[i].x, f.mesh3[i].y, f.mesh3[i].z);
    };
    auto unit3 = [](cv::Vec3f v) {
        const float n = std::sqrt(v.dot(v));
        return n > 1e-6f ? v * (1.f / n) : v;
    };

    // The unit is the distance between the eye *centres*, not the outer
    // corners: every proportion below and every constant in face3d is expressed
    // in eye separations, and the outer corners are about 1.45x that, which
    // would scale the whole rig up by the same factor.
    auto eyeCentre = [&](int a, int b, int c, int d) {
        return (V(a) + V(b) + V(c) + V(d)) * 0.25f;
    };
    const cv::Vec3f cR = eyeCentre(kEyeROuter, kEyeRInner, kEyeRUp, kEyeRLow);
    const cv::Vec3f cL = eyeCentre(kEyeLOuter, kEyeLInner, kEyeLUp, kEyeLLow);
    const cv::Vec3f span = cL - cR;
    const float unit = std::sqrt(span.dot(span));
    if (unit < 12.f) return;

    cv::Vec3f ex = unit3(span);
    const cv::Vec3f ey0 = unit3(V(kChin) - V(kForehead));
    // ez completes a right-handed frame; with image y pointing down, that
    // points *away* from the camera, which is the depth direction wanted.
    cv::Vec3f ez = unit3(ex.cross(ey0));
    const cv::Vec3f ey = unit3(ez.cross(ex)); // re-orthogonalise
    // MediaPipe labels by the subject's anatomy, so on a mirrored preview the
    // "left" outer eye corner is on the image right; flip so +x is image-right.
    if (ex[0] < 0.f) { ex = -ex; ez = -ez; }

    face3d::Head h;
    h.R = cv::Matx33f(ex[0], ey[0], ez[0],
                      ex[1], ey[1], ez[1],
                      ex[2], ey[2], ez[2]); // columns: right, down, back
    h.unit = unit;
    const cv::Point2f eyeMid = (f.eyeL + f.eyeR) * 0.5f - off;
    h.anchor = eyeMid;
    h.phase = phase;

    // Proportions, read in the head's own frame -- exact here, because the
    // frame is 3D and nothing is foreshortened.
    const cv::Vec3f origin = (cR + cL) * 0.5f; // matches h.anchor exactly
    auto inHead = [&](int i) {
        const cv::Vec3f d = V(i) - origin;
        return cv::Vec3f(d.dot(ex), d.dot(ey), d.dot(ez)) * (1.f / unit);
    };
    const cv::Vec3f crown = inHead(kForehead);
    const cv::Vec3f nose = inHead(kNoseTip);
    const cv::Vec3f tL = inHead(kTempleR), tR = inHead(kTempleL);
    h.crownY = clampf(crown[1], -1.60f, -0.45f);
    h.headHalfW = clampf(0.5f * std::fabs(tR[0] - tL[0]), 0.70f, 1.80f);
    h.noseY = clampf(nose[1], 0.35f, 1.10f);
    h.noseZ = clampf(nose[2], -0.90f, -0.05f);

    face3d::render(frame, h, species);
}

void FaceFilter::applyAnimalFace(cv::Mat& frame, const Face& f,
                                 cv::Point2f off, double phase,
                                 face3d::Species species) const {
    if ((int)f.mesh.size() < 468) return;
    const float eyeSep = len(f.eyeR - f.eyeL);
    if (eyeSep < 8.f) return;

    cv::Rect b(f.box.x - (int)off.x - 2, f.box.y - (int)off.y - 2,
               f.box.width + 4, f.box.height + 4);
    b &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (b.width < 8 || b.height < 8) return;

    // Straight onto the frame: the tessellation tiles the face without gaps,
    // so the paint is opaque anyway and the scratch copy it used to accumulate
    // into was pure cost -- a region-sized allocation and two extra passes over
    // every pixel, every frame.
    cv::Mat ov = frame(b);
    const cv::Point org = b.tl();
    const cv::Point2f eyeMid = (f.eyeL + f.eyeR) * 0.5f - off;
    const float invEye = 1.f / eyeSep;

    // Colour every vertex once, from where it sits in the head's frame, then
    // let the triangles interpolate between them. Evaluating per vertex rather
    // than per triangle is what makes the markings smooth across the mesh.
    const bool isPig = (species == face3d::Species::Pig);
    const bool isGrinch = (species == face3d::Species::Grinch);
    const float furDepth = isGrinch ? kGrinchFur : (isPig ? kPigFur : kDogFur);
    const float furCrown = isGrinch ? 1.40f : 0.f;

    std::vector<cv::Vec3f> vcol(f.mesh.size());
    std::vector<cv::Point2f> vpos(f.mesh.size());
    std::vector<cv::Point2f> vuv(f.mesh.size());
    for (size_t i = 0; i < f.mesh.size(); ++i) {
        const cv::Point2f q = f.mesh[i] - off;
        vpos[i] = cv::Point2f(q.x - org.x, q.y - org.y);
        const cv::Point2f d = q - eyeMid;
        vuv[i] = cv::Point2f(dot(d, f.right) * invEye, dot(d, f.down) * invEye);
        vcol[i] = isGrinch ? grinchColourAt(vuv[i].x, vuv[i].y)
                  : isPig  ? pigColourAt(vuv[i].x, vuv[i].y)
                           : dogColourAt(vuv[i].x, vuv[i].y);
    }

    // Mean brightness of the face, so the shading term below is relative to
    // this subject in this light rather than to an assumed exposure. Sampled
    // on a coarse grid: it only has to be approximately right.
    double lumaSum = 0.0;
    int lumaN = 0;
    for (int y = 0; y < ov.rows; y += 4) {
        const cv::Vec3b* r = ov.ptr<cv::Vec3b>(y);
        for (int x = 0; x < ov.cols; x += 4) {
            lumaSum += 0.114f * r[x][0] + 0.587f * r[x][1] + 0.299f * r[x][2];
            ++lumaN;
        }
    }
    const float meanLuma = lumaN ? (float)(lumaSum / lumaN) : 128.f;
    const float invMeanLuma = 1.f / std::max(16.f, meanLuma);

    const auto& tess = MpConn::kFaceLandmarksTesselation;
    for (int t = 0; t < kMeshTriangles; ++t) {
        const int ia = tess[3 * t][0], ib = tess[3 * t + 1][0],
                  ic = tess[3 * t + 2][0];
        const cv::Point2f tri[3] = {vpos[ia], vpos[ib], vpos[ic]};
        const cv::Vec3f col[3] = {vcol[ia], vcol[ib], vcol[ic]};
        const cv::Point2f uv[3] = {vuv[ia], vuv[ib], vuv[ic]};
        fillTriangleSmooth(ov, tri, col, uv, invMeanLuma, furDepth, furCrown);
    }
    // Ears and nose on top, as real geometry. They cannot come from the face
    // mesh -- it stops at the face -- so they are oriented by a basis measured
    // from it in 3D instead.
    drawAnimalParts(frame, f, off, phase, species);
}

Filter nextFilter(Filter f) {
    switch (f) {
        case Filter::None:     return Filter::BigSmile;
        case Filter::BigSmile: return Filter::Crying;
        case Filter::Crying:   return Filter::FaceMesh;
        case Filter::FaceMesh: return Filter::DogFace;
        case Filter::DogFace:  return Filter::PigFace;
        case Filter::PigFace:  return Filter::Grinch;
        case Filter::Grinch:   return Filter::None;
    }
    return Filter::None;
}

const char* filterName(Filter f) {
    switch (f) {
        case Filter::None:     return "Filter Off";
        case Filter::BigSmile: return "Big Smile";
        case Filter::Crying:   return "Crying";
        case Filter::FaceMesh: return "Face Mesh";
        case Filter::DogFace:  return "Dog Face";
        case Filter::PigFace:  return "Pig Face";
        case Filter::Grinch:   return "Grinch";
    }
    return "";
}

} // namespace olc
