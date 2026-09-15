#pragma once

// Shared enums and small value types used across the UI and app modules.

namespace olc {

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
    StartCamera, // welcome -> live camera
    Sleep,       // welcome -> blank the screen (display sleep)
    Home,        // camera -> welcome screen
    Quit,
};

// Live facial-expression filter applied to the camera preview (and captures).
// The face is reshaped in place rather than having graphics drawn over it; only
// the crying filter's tears are drawn on top.
enum class Filter {
    None,
    BigSmile, // mouth stretched into a wide grin; teeth pop when it opens
    Crying,   // mouth/brows pulled into a frown, with falling tears
};

} // namespace olc
