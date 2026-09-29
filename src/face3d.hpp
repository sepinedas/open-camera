#pragma once

#include <opencv2/core.hpp>

#include "types.hpp"

// Three-dimensional ears and muzzles for the face-paint filters.
//
// Their *markings* are painted onto the MediaPipe face mesh (see
// FaceFilter::applyAnimalFace), which is what makes them sit on the skin and
// deform with it. Ears and a muzzle cannot come from that mesh -- it ends at
// the face, and a nose has to stand off it -- so they are real triangle
// meshes, shaded and z-buffered, drawn over the painted face.
//
// Unlike a sticker, they are oriented by a basis measured from the face mesh
// *in three dimensions*: MediaPipe gives every landmark a depth, so the head's
// axes can be read straight off it rather than inferred from how features sit
// inside a detection box. Everything below is expressed in that frame, in
// units of one eye separation.
namespace olc::face3d {

// Whose ears and muzzle to build. The rig around them -- the basis measured
// off the face mesh, the perspective-free projection, the z-buffer, the
// shading -- is identical, so they differ only by a table of geometry and
// colours (see Style in the .cpp).
enum class Species {
    Dog,
    Pig,
    Grinch,
    // The odd one out: not ears and a muzzle added over a painted face, but a
    // whole head in place of it -- skull, hinged jaw, teeth and all. It needs
    // nothing painted underneath, so the filter that draws it skips the mesh
    // paint entirely and calls render() on its own.
    Shark,
};

// The head's frame, measured from the face mesh, in image space.
struct Head {
    // Columns are the head's right, down and backward axes as unit vectors in
    // (x pixels, y pixels, z depth). R * p therefore maps a model-space offset
    // onto an image offset plus a depth.
    cv::Matx33f R;
    cv::Point2f anchor;   // where the model origin (the eye midpoint) lands
    float unit = 0.f;     // pixels per eye separation; the model's scale

    // Where this face's features are, in eye-separation units in the head's
    // frame: +x toward the image-right eye, +y toward the chin, +z away from
    // the camera. Measured from the mesh, so the ears sit on the real
    // silhouette and the nose on the real nose.
    float headHalfW = 1.15f; // half the head width at the temples
    float crownY = -0.90f;   // top of the head, above the eye line
    float noseY = 0.65f;     // nose tip, below the eye line
    float noseZ = -0.35f;    // how far the nose tip stands out of the face
    float chinY = 1.35f;     // bottom of the chin. Only the shark needs it,
                             // which is sized to the whole head rather than
                             // hung off one feature of it.

    // What the face is doing. A model that replaces the head has to move with
    // the face or it is a mask sitting on top of one, so the shark works its
    // jaw, narrows its eyes, curls its gape and puts its tongue out from
    // these; the animals tilt their ears and loll a tongue.
    Expression expr;

    double phase = 0.0; // free-running frame counter; drives the ear wiggle
};

// Draw the ears and nose over `frame` (BGR, 8-bit). A no-op when the head is
// too small to render cleanly.
void render(cv::Mat& frame, const Head& head, Species species);

// The frame-space rectangle render() will touch for this head, exactly -- the
// projected bounding box of the model, not an estimate. Callers that have to
// prepare a region before drawing into it (the NV12 preview path converts and
// re-encodes only what changes) use this instead of padding the face box by a
// worst case. Empty when the head is too small to draw.
cv::Rect bounds(const Head& head, Species species);

} // namespace olc::face3d
