#pragma once

#include <opencv2/core.hpp>

// Three-dimensional ears and nose for the dog filter.
//
// The dog's *markings* are painted onto the MediaPipe face mesh (see
// FaceFilter::applyDogFace), which is what makes them sit on the skin and
// deform with it. Ears and a nose cannot come from that mesh -- it ends at the
// face, and a nose has to stand off it -- so they are real triangle meshes,
// shaded and z-buffered, drawn over the painted face.
//
// Unlike a sticker, they are oriented by a basis measured from the face mesh
// *in three dimensions*: MediaPipe gives every landmark a depth, so the head's
// axes can be read straight off it rather than inferred from how features sit
// inside a detection box. Everything below is expressed in that frame, in
// units of one eye separation.
namespace olc::dog3d {

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

    double phase = 0.0; // free-running frame counter; drives the ear wiggle
};

// Draw the ears and nose over `frame` (BGR, 8-bit). A no-op when the head is
// too small to render cleanly.
void render(cv::Mat& frame, const Head& head);

} // namespace olc::dog3d
