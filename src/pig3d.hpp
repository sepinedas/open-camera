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
namespace olc::pig3d {

// What the caller knows about the head this frame. The eye centres fix the
// in-plane orientation (roll) and the scale; `yaw`/`pitch` are the out-of-plane
// orientation. `filters` measures all four from the MediaPipe face mesh; when a
// field is unavailable the renderer falls back to a front-facing, upright head
// sized from the face box.
struct Head {
    bool hasEyes = false;
    cv::Point2f leftEye{-1.f, -1.f};  // image-left eye centre
    cv::Point2f rightEye{-1.f, -1.f}; // image-right eye centre
    // True when `yaw`/`pitch` were measured; otherwise the renderer estimates
    // them from where the eyes sit inside the face box (the old heuristic).
    bool hasPose = false;
    float yaw = 0.f;   // radians, + => head turned toward image-right
    float pitch = 0.f; // radians, + => chin up
};

// Draw the 3D pig over `frame` (BGR, 8-bit) for one face.
//
//   face   face bounding box, image coords (roi-local when called per-region)
//   head   measured head landmarks/pose (see above)
//   phase  free-running frame counter; drives a subtle ear wiggle
void render(cv::Mat& frame, const cv::Rect& face, const Head& head, double phase);

// Convenience overload for callers that only have the two eye centres (the
// mockup/preview tools). Pass hasEyes=false to orient from the box alone.
void render(cv::Mat& frame, const cv::Rect& face, bool hasEyes,
            cv::Point2f leftEye, cv::Point2f rightEye, double phase);

} // namespace olc::pig3d
