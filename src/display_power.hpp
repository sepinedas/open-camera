#pragma once

#include <string>
#include <vector>

struct SDL_Window;

namespace olc {

// Puts the screen to sleep as deeply as this machine allows, and wakes it.
//
// A black frame is not sleep: an LCD's backlight is most of its power draw and
// stays lit behind black pixels, and the display pipeline keeps scanning the
// frame out sixty times a second. So sleep switches off whatever is there to
// switch off, each step independent of the others and none of them required:
//
//   * the backlight, through the kernel's backlight class
//     (/sys/class/backlight/*) -- the HyperPixel's GPIO backlight, the
//     official 7" and other DSI panels' controllers;
//   * the display pipeline itself (DPMS), through the DRM device SDL already
//     holds, which stops scanout and lets the panel driver power the panel
//     down, and puts an HDMI monitor into standby. Needs libdrm at build time
//     (OLC_HAVE_LIBDRM) and SDL's kmsdrm driver at run time;
//   * failing both, `vcgencmd display_power`, for the legacy firmware display
//     stack, where the other two do not exist.
//
// Everything switched off is remembered and restored by wake().
class DisplayPower {
public:
    // Switch the display off. Returns what was switched off, for the log;
    // empty when nothing could be, and the screen is merely black.
    std::string sleep(SDL_Window* win);
    void wake(SDL_Window* win);

    // True when something was actually switched off, so there is no point
    // presenting frames until wake().
    bool dark() const { return !backlights_.empty() || dpms_ || vcgencmd_; }

private:
    struct Backlight {
        std::string dir;        // /sys/class/backlight/<name>
        std::string brightness; // as it was before sleep
        std::string blPower;
    };
    std::vector<Backlight> backlights_;
    bool dpms_ = false;
    bool vcgencmd_ = false;
};

} // namespace olc
