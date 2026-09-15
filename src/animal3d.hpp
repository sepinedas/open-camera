#pragma once

#include <opencv2/core.hpp>

// A tiny software 3D renderer for the pig-face filter.
//
// Rather than pasting flat sprites over the face, this builds real 3D meshes for
// the pig's ears and snout, estimates the head's pose (roll / yaw / pitch) from
// the face landmarks, and renders the meshes through a perspective camera with a
// z-buffer, per-pixel (Gouraud) shading and supersampled anti-aliasing. Because
// the geometry is genuinely three-dimensional, the snout protrudes and
// foreshortens, and the ears swing around and occlude behind the head exactly as
// the head turns -- they share the face's orientation and perspective instead of
// looking like decals.
namespace olc::animal3d {

// Which animal to build. The rig -- head pose, perspective camera, z-buffer,
// shading -- is identical for both; only the muzzle and ear geometry and the
// colours differ, so they are described by data rather than by a second
// renderer (see Style in the .cpp).
enum class Species { Pig, Dog };

// Where this particular face's features actually sit, measured from the
// MediaPipe mesh. All values are in *eye-separation units* in the head's own
// frame, with +y pointing down from the eye midpoint -- the same space the
// pig meshes are modelled in, so they compose directly with the rig.
//
// Measuring these is what stops the pig being a fixed mask scaled by eye
// distance. The defaults below are the constants the meshes used to hardcode;
// they are kept as the fallback for callers with no landmarks (the preview
// tool). They are exactly the old hardcoded values, so a caller that measures
// nothing gets the previous pig unchanged and any visible difference is
// attributable to the measurement rather than to a shifted default. Note
// `noseY`: a real nose tip sits about 0.65 eye-separations below the eye line,
// so this 0.32 is what used to put the snout up on the bridge.
struct Proportions {
    float noseY = 0.32f;      // nose tip, below the eye line
    float noseHalfW = 0.28f;  // half the alar (nostril) width
    float crownY = -0.90f;    // top of the head, above the eye line
    float headHalfW = 1.22f;  // half the head width at the temples
};

// What the caller knows about the head this frame. The eye centres fix the
// in-plane orientation (roll) and the scale; `yaw`/`pitch` are the out-of-plane
// orientation. `filters` measures all of it from the MediaPipe face mesh; when
// a field is unavailable the renderer falls back to a front-facing, upright
// head sized from the face box.
struct Head {
    bool hasEyes = false;
    cv::Point2f leftEye{-1.f, -1.f};  // image-left eye centre
    cv::Point2f rightEye{-1.f, -1.f}; // image-right eye centre
    // True when `yaw`/`pitch` were measured; otherwise the renderer estimates
    // them from where the eyes sit inside the face box (the old heuristic).
    bool hasPose = false;
    float yaw = 0.f;   // radians, + => head turned toward image-right
    float pitch = 0.f; // radians, + => chin up
    // Where this face's nose and head edges actually are. Left at the defaults
    // the pig is the old one-size rig.
    Proportions prop;
};

// Draw the 3D pig over `frame` (BGR, 8-bit) for one face.
//
//   face   face bounding box, image coords (roi-local when called per-region)
//   head   measured head landmarks/pose (see above)
//   phase  free-running frame counter; drives a subtle ear wiggle
void render(cv::Mat& frame, const cv::Rect& face, const Head& head, double phase,
            Species species = Species::Pig);

// Convenience overload for callers that only have the two eye centres (the
// mockup/preview tools). Pass hasEyes=false to orient from the box alone.
void render(cv::Mat& frame, const cv::Rect& face, bool hasEyes,
            cv::Point2f leftEye, cv::Point2f rightEye, double phase,
            Species species = Species::Pig);

} // namespace olc::animal3d
