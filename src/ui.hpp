#pragma once

#include <vector>

#include <SDL2/SDL.h>

#include "types.hpp"

namespace olc {

// One circular, icon-only, translucent control.
struct Button {
    Action action;
    int cx, cy, r;
};

// Manages the auto-hiding, translucent menu: it tracks the last interaction and
// derives a fade alpha, lays out the icon buttons for the current mode, draws
// them, and hit-tests taps.
class Menu {
public:
    // Register a tap/keypress: brings the menu fully back and restarts the timer.
    void wake() { lastActivity_ = SDL_GetTicks(); }

    // Current fade opacity (0..255). Falls to 0 a few seconds after the last
    // interaction; a wake() snaps it back to full.
    Uint8 alpha() const;

    // True while any part of the menu is still visible.
    bool awake() const { return alpha() > 0; }

    // Build the button set for a mode. `hasVideo` adds a Play button in the
    // gallery only when the selected item is a video; `canSwitchCamera` adds
    // the camera-switch button to the camera row only when a second camera is
    // attached. Both flags must match between drawing and hit-testing, or the
    // row is laid out differently for the two -- see App::buttonsFor().
    std::vector<Button> layout(Mode mode, int screenW, int screenH,
                               bool hasVideo, bool canSwitchCamera = false) const;

    // Draw one button (translucent disc + icon) at the given menu alpha.
    static void drawButton(SDL_Renderer* ren, const Button& b, Uint8 alpha);

    // Return the action of the button under (x, y), or Action::None.
    static Action hitTest(const std::vector<Button>& buttons, int x, int y);

private:
    Uint32 lastActivity_ = 0;
};

} // namespace olc
