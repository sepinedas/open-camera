#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "types.hpp"

namespace olc {

// Real-time, WhatsApp-style facial-expression filters.
//
// The expression filters *reshape* the face itself (its pixels are pushed
// around with cv::remap) rather than pasting cartoon graphics over it -- a "big
// smile" is your own mouth stretched into a grin, a "crying" face is your own
// mouth and brows pulled into a frown, the only thing drawn on top being the
// crying tears.
//
// The "pig face" filter instead overlays real 3D models -- mesh ears and a
// protruding snout with nostrils -- rendered by `pig3d` through a perspective
// camera, oriented by the head pose so the snout foreshortens and the ears
// swing around the head instead of sitting on top like stickers.
//
// Faces are found with **MediaPipe's Face Landmarker** (Tasks Vision C++ API,
// CPU/TFLite), which returns a dense 478-point face mesh per face plus the
// blendshape scores. That is what every filter is driven from:
//
//   * the warps are anchored to the *actual* mouth corners, lips and eyebrows
//     instead of fixed fractions of a detector box, so they follow the real
//     mouth wherever it is and whatever shape it already has;
//   * mouth openness comes from the `jawOpen` blendshape rather than being
//     guessed from the contrast of a mouth-shaped patch;
//   * head roll/yaw/pitch for the pig are measured from the eye, nose, cheek
//     and chin landmarks instead of from where a pair of eye boxes happen to
//     sit inside a face box.
//
// The MediaPipe model bundle (`face_landmarker.task`) is found in the usual
// install locations, or pointed at explicitly with `--face-model`.
//
// The MediaPipe headers pull in Abseil/protobuf/TFLite and need C++20, so the
// landmarker lives behind a pimpl: only filters.cpp sees them.
class FaceFilter {
public:
    // Loads the Face Landmarker bundle from the usual install locations.
    FaceFilter();
    ~FaceFilter();
    FaceFilter(const FaceFilter&) = delete;
    FaceFilter& operator=(const FaceFilter&) = delete;

    // Point the landmarker at an explicit model bundle (from --face-model).
    // Empty is a no-op; a bad path leaves any already-loaded model in place.
    void setModel(const std::string& path);

    // True once a model is loaded and filtering can actually do something.
    bool ready() const;

    // Apply `filter` to `frame` (BGR, 8-bit, 3-channel) in place. `phase` is a
    // free-running per-frame counter that drives the tear animation. A no-op
    // when the filter is None, no model loaded, or no face is found.
    void apply(cv::Mat& frame, Filter filter, double phase);

    // --- region-limited API (keeps the NV12 preview off the CPU convert) ---
    //
    // Refresh the detected faces from `src`, a BGR frame or a plain
    // luma/grayscale one. Call once per frame: inference runs every frame, and
    // MediaPipe's VIDEO mode tracks the face between full detections itself,
    // so there is no detect-every-N-frames throttle on top.
    //
    // `src` may be any size -- it is downscaled to detectionWidth() internally
    // -- but a caller that can produce the detection image cheaply (see
    // Camera::nv12ToBGRScaled) should pre-scale it to exactly that width.
    void updateDetection(const cv::Mat& src);

    // Width the detection image is scaled to before inference. Landmarks come
    // back normalized, so this never affects the coordinates -- only how much
    // detail small faces keep.
    static int detectionWidth();

    // The single frame-space rectangle covering everything `filter` will modify
    // for the currently-detected faces (face boxes + margin for the warp and
    // tears, clamped to WxH and made even for chroma-subsampled buffers). An
    // empty rect means there is nothing to reshape this frame.
    cv::Rect dirtyRegion(Filter filter, int w, int h) const;

    // Apply `filter` to `roi`, a BGR sub-image whose top-left sits at `origin`
    // in frame space. Only the parts of each face falling inside `roi` are
    // touched, so callers can convert and re-encode just the dirty region.
    void applyRegion(cv::Mat& roi, cv::Point origin, Filter filter, double phase);

    // Draw the 3D pig-face graphics for an explicit face box and eye landmarks,
    // bypassing detection. Pass eye centres (image coords) to orient it; pass
    // (-1,-1) for either to fall back to the box (upright). Used by the mockup
    // tools and tests to preview the effect deterministically.
    void drawPigPreview(cv::Mat& frame, const cv::Rect& face, cv::Point2f leftEye,
                        cv::Point2f rightEye, double phase) const;

private:
    // Everything the filters need to know about one detected face, in full-res
    // frame coordinates. Distilled from a MediaPipe FaceLandmarkerResult: the
    // handful of mesh points each filter is anchored to, resolved into *image*
    // order (left/right as seen on screen, so a mirrored preview works too),
    // plus the head pose and expression measured from the mesh.
    struct Face {
        cv::Rect box;               // tight bounding box of the whole mesh
        cv::Point2f eyeL, eyeR;     // eye centres
        cv::Point2f mouthL, mouthR; // mouth corners
        cv::Point2f lipTop, lipBot; // inner-lip centres (upper / lower)
        cv::Point2f browL, browR;   // inner eyebrow ends
        cv::Point2f lidL, lidR;     // lower-eyelid centres: where tears well up
        cv::Point2f right, down;    // unit vectors along / across the eye line
        float yaw = 0.f;            // radians, + => head turned toward image-right
        float pitch = 0.f;          // radians, + => chin up
        float open = 0.f;           // 0..1 how far the jaw is open
        float smile = 0.f;          // 0..1 how much the mouth already grins
    };

    // MediaPipe landmarker + the scratch buffers it needs; defined in the .cpp
    // so the MediaPipe headers stay out of everything that includes this file.
    struct Landmarker;

    // Run the landmarker on `src` (CV_8UC1 luma or CV_8UC3 BGR) and rebuild
    // `faces_` from the mesh it returns. Callers go through updateDetection(),
    // which owns the model-missing warning.
    void detect(const cv::Mat& src);

    void applySmile(cv::Mat& frame, const Face& f, cv::Point2f off) const;
    void applyCry(cv::Mat& frame, const Face& f, cv::Point2f off, double phase) const;
    // Draw the smooth 3D pig ears/snout over one face, oriented by its measured
    // head pose. `phase` drives a gentle ear wiggle.
    void applyPig(cv::Mat& frame, const Face& f, cv::Point2f off, double phase) const;

    // Brighten toward white the teeth showing between the lips, given the
    // mouth's *post-warp* corners and inner-lip centres. Stronger the wider
    // the grin opens.
    void whitenTeeth(cv::Mat& frame, cv::Point2f cornerL, cv::Point2f cornerR,
                     cv::Point2f lipTop, cv::Point2f lipBot, float strength) const;
    // Draw the falling tears of the crying filter.
    void drawTears(cv::Mat& frame, const Face& f, cv::Point2f off, double phase) const;

    std::unique_ptr<Landmarker> lm_;
    bool warned_ = false;     // "no model" logged only once
    std::vector<Face> faces_; // last detection result, full-res coords
};

// Search the usual install locations for MediaPipe's `face_landmarker.task`
// bundle (including the versioned /opt/mediapipe/<ver> layout the
// media-pipe-builder .deb uses). Returns an empty string when none is found.
// Exposed for reuse/testing.
std::string findFaceLandmarkerModel();

// Cycle order for the on-screen filter button: None -> BigSmile -> Crying ->.
Filter nextFilter(Filter f);
// Short human label for the brief on-screen filter name.
const char* filterName(Filter f);

} // namespace olc
