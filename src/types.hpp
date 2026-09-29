#pragma once

// Shared enums and small value types used across the UI and app modules.

namespace olc {

// What the face is doing, each 0..1 unless noted.
//
// Taken from MediaPipe's blendshapes, which are the ARKit-style set, with a
// measurement off the mesh as a fallback where one is reliable -- eyelid
// aperture is a distance between two landmarks and needs no model to read,
// and a bundle without the blendshape head is still usable that way.
//
// Sides are the IMAGE's, not the subject's: MediaPipe labels by anatomy, so
// which of them lands on screen-left depends on whether the preview is
// mirrored, and detect() settles that once for every pair.
struct Expression {
    float jawOpen = 0.f;
    float smile = 0.f;    // corners pulled up and back
    float frown = 0.f;    // and down
    float blinkL = 0.f;   // 1 = shut. Image-left eye
    float blinkR = 0.f;
    float browUp = 0.f;   // raised, inner and outer together
    float browDown = 0.f; // lowered: the scowl
    float pucker = 0.f;   // lips pushed forward
    float cheekPuff = 0.f;
    float tongue = 0.f;   // see the note on tongueOut in filters.cpp
    float jawSide = 0.f;  // -1 fully image-left, +1 fully image-right
};

// The high-level screen the app is currently on.
enum class Mode {
    Welcome,       // start screen: Lego-brick camera + Start / Sleep options
    Camera,        // live preview + capture controls
    Gallery,       // browse captured photos/videos
    Playback,      // playing a video from the gallery
    ConfirmDelete, // icon-only yes/no before deleting
    Sleep,         // display blanked (screen off); double-tap to wake
};

// Every tappable control maps to exactly one action.
enum class Action {
    None,
    Shutter,     // take a photo
    ZoomIn,
    ZoomOut,
    OpenGallery, // camera -> gallery
    Back,        // gallery -> camera
    Prev,        // previous item in gallery
    Next,        // next item in gallery
    Play,        // play the selected video
    Delete,      // ask to delete the selected item
    ConfirmYes,  // confirm deletion
    ConfirmNo,   // cancel deletion
    CycleFilter, // cycle the live facial-expression filter
    SwitchCamera, // switch to the next camera (Pi camera <-> USB webcam)
    StartCamera, // welcome -> live camera
    Sleep,       // welcome -> blank the screen (display sleep)
    Home,        // camera -> welcome screen
    Quit,
};

// Live filter applied to the camera preview (and captures). The expression
// filters reshape the face in place rather than having graphics drawn over it;
// only the crying filter's tears, and the face-mesh overlay, are drawn on top.
enum class Filter {
    None,
    BigSmile, // mouth stretched into a wide grin; teeth pop when it opens
    Crying,   // mouth/brows pulled into a frown, with falling tears
    FaceMesh, // the tracked landmarks drawn over the face as dots and edges
    DogFace,  // a dog painted onto the face mesh, so it moves with the face
    PigFace,  // the same, built as a pig: snout and upright ears
    Grinch,   // green, shaggy, pointed ears and a heavy scowl
    Shark,    // the whole head replaced by a 3D shark, jaw and all
};

} // namespace olc
