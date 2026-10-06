#include "app.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <iostream>
#include <sys/stat.h>
#include <vector>

#include <SDL2/SDL2_gfxPrimitives.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "icons.hpp"

namespace olc {

// How far a photo in the gallery can be pinched in. Further than the live
// camera's digital zoom: a still is full resolution, so there is detail to
// find, and nothing has to be re-encoded at that size every frame.
static constexpr double kGalleryMaxZoom = 6.0;

// Rotate a BGR frame clockwise by `deg` (0/90/180/270). Returns a rotated copy;
// for 0 (or any non-multiple) it returns the input unchanged. Used to bake the
// camera-image rotation into captured stills so they match the preview.
static cv::Mat rotatedBGR(const cv::Mat& in, int deg) {
    switch (((deg % 360) + 360) % 360) {
        case 90:  { cv::Mat o; cv::rotate(in, o, cv::ROTATE_90_CLOCKWISE); return o; }
        case 180: { cv::Mat o; cv::rotate(in, o, cv::ROTATE_180); return o; }
        case 270: { cv::Mat o; cv::rotate(in, o, cv::ROTATE_90_COUNTERCLOCKWISE); return o; }
        default:  return in;
    }
}

App::~App() {
    if (tex_) SDL_DestroyTexture(tex_);
    if (canvas_) SDL_DestroyTexture(canvas_);
    if (thumbTex_) SDL_DestroyTexture(thumbTex_);
    if (ren_) SDL_DestroyRenderer(ren_);
    if (win_) SDL_DestroyWindow(win_);
    SDL_Quit();
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

bool App::initDisplay() {
    // Decide which SDL video drivers to try, in order.
    std::vector<std::string> drivers;
    if (!cfg_.driver.empty()) {
        drivers = {cfg_.driver};
    } else if (std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY")) {
        drivers = {""}; // a desktop session is present: let SDL auto-pick
    } else {
        drivers = {"kmsdrm", "fbcon"}; // headless Pi: draw straight to HDMI
    }

    Uint32 winFlags = cfg_.windowed ? 0u : (Uint32)SDL_WINDOW_FULLSCREEN_DESKTOP;
    int w = cfg_.windowed ? cfg_.width : 0;
    int h = cfg_.windowed ? cfg_.height : 0;

    for (const std::string& drv : drivers) {
        const char* tag = drv.empty() ? "(auto)" : drv.c_str();
        if (drv.empty()) unsetenv("SDL_VIDEODRIVER");
        else setenv("SDL_VIDEODRIVER", drv.c_str(), 1);

        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
            std::cerr << "display: driver '" << tag << "' SDL_Init failed: "
                      << SDL_GetError() << "\n";
            continue;
        }

        // Report what KMS/DRM outputs SDL can see -- the key clue when a screen
        // stays black (0 displays = no connected HDMI/DPI connector with a mode).
        int nd = SDL_GetNumVideoDisplays();
        std::cerr << "display: driver '" << tag << "' up; " << nd
                  << " output(s) detected\n";
        for (int i = 0; i < nd; ++i) {
            SDL_Rect b{};
            SDL_GetDisplayBounds(i, &b);
            const char* name = SDL_GetDisplayName(i);
            std::cerr << "  [" << i << "] " << (name ? name : "?") << " "
                      << b.w << "x" << b.h << "\n";
        }

        win_ = SDL_CreateWindow("open-lego-camera", SDL_WINDOWPOS_CENTERED,
                                SDL_WINDOWPOS_CENTERED,
                                cfg_.windowed ? w : 0, cfg_.windowed ? h : 0,
                                winFlags);
        if (!win_) {
            std::cerr << "display: driver '" << tag << "' CreateWindow failed: "
                      << SDL_GetError() << "\n";
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
            SDL_Quit();
            continue;
        }

        ren_ = SDL_CreateRenderer(win_, -1,
                                  SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!ren_) ren_ = SDL_CreateRenderer(win_, -1, 0); // software fallback
        if (!ren_) {
            std::cerr << "display: driver '" << tag << "' CreateRenderer failed: "
                      << SDL_GetError() << "\n";
            SDL_DestroyWindow(win_);
            win_ = nullptr;
            SDL_Quit();
            continue;
        }

        SDL_GetRendererOutputSize(ren_, &screenW_, &screenH_);
        SDL_ShowCursor(SDL_DISABLE);
        SDL_SetRenderDrawBlendMode(ren_, SDL_BLENDMODE_BLEND);

        // Set up UI rotation. For 90/270 the logical canvas is the panel with
        // width/height swapped; everything is drawn there and blitted rotated.
        rotate_ = cfg_.rotate;
        if (rotate_ != 0 && !SDL_RenderTargetSupported(ren_)) {
            std::cerr << "display: renderer can't rotate (no target textures); "
                         "drawing unrotated\n";
            rotate_ = 0;
        }
        bool swap = (rotate_ == 90 || rotate_ == 270);
        viewW_ = swap ? screenH_ : screenW_;
        viewH_ = swap ? screenW_ : screenH_;
        if (rotate_ != 0) {
            canvas_ = SDL_CreateTexture(ren_, SDL_PIXELFORMAT_ARGB8888,
                                        SDL_TEXTUREACCESS_TARGET, viewW_, viewH_);
            if (!canvas_) {
                std::cerr << "display: canvas alloc failed (" << SDL_GetError()
                          << "); drawing unrotated\n";
                rotate_ = 0;
                viewW_ = screenW_;
                viewH_ = screenH_;
            } else {
                SDL_SetTextureBlendMode(canvas_, SDL_BLENDMODE_NONE);
            }
        }

        const char* used = SDL_GetCurrentVideoDriver();
        std::cout << "display: " << (used ? used : "?") << " " << screenW_
                  << "x" << screenH_;
        if (rotate_) std::cout << " (UI rotated " << rotate_ << ", logical "
                               << viewW_ << "x" << viewH_ << ")";
        std::cout << "\n";
        return true;
    }

    std::cerr << "could not open a display. On a headless Pi run this on the "
                 "active HDMI console (not SSH), or pass --driver. See the "
                 "\"Debugging HDMI / no display\" section in the README.\n";
    return false;
}

void App::clear() {
    SDL_SetRenderDrawColor(ren_, 0, 0, 0, 255);
    SDL_RenderClear(ren_);
}

// Point every subsequent draw call at the logical canvas (when rotating).
void App::beginFrame() {
    if (canvas_) SDL_SetRenderTarget(ren_, canvas_);
}

// Finish the frame: blit the logical canvas onto the panel, rotated, and flip.
void App::present() {
    if (canvas_) {
        SDL_SetRenderTarget(ren_, nullptr);
        SDL_SetRenderDrawColor(ren_, 0, 0, 0, 255);
        SDL_RenderClear(ren_);
        // A dst rect the size of the logical canvas, centred on the panel, then
        // rotated: for 90/270 its bounding box becomes the full panel.
        SDL_Rect dst{(screenW_ - viewW_) / 2, (screenH_ - viewH_) / 2,
                     viewW_, viewH_};
        SDL_RenderCopyEx(ren_, canvas_, nullptr, &dst, (double)rotate_, nullptr,
                         SDL_FLIP_NONE);
    }
    SDL_RenderPresent(ren_);
}

// Undo the display rotation so a panel tap lands on the right UI element.
void App::physicalToView(int px, int py, int& vx, int& vy) const {
    switch (rotate_) {
        case 90:  vx = py;             vy = screenW_ - px; break;
        case 180: vx = screenW_ - px;  vy = screenH_ - py; break;
        case 270: vx = screenH_ - py;  vy = px;            break;
        default:  vx = px;             vy = py;            break;
    }
    vx = std::min(viewW_ - 1, std::max(0, vx));
    vy = std::min(viewH_ - 1, std::max(0, vy));
}

// Scaled bitmap text via SDL2_gfx's 8x8 font: render once into a small target
// texture, then blit it magnified. Keeps text crisp and avoids an SDL_ttf/font
// dependency. Falls back to unscaled text if target textures aren't supported.
void App::drawText(int x, int topY, const std::string& s, int scale,
                   SDL_Color c, bool center) {
    if (s.empty()) return;
    int w = 8 * (int)s.size(), h = 8;
    SDL_Texture* prev = SDL_GetRenderTarget(ren_);
    SDL_Texture* t = SDL_CreateTexture(ren_, SDL_PIXELFORMAT_ARGB8888,
                                       SDL_TEXTUREACCESS_TARGET, w, h);
    if (!t) {
        stringRGBA(ren_, (Sint16)(center ? x - w / 2 : x), (Sint16)topY,
                   s.c_str(), c.r, c.g, c.b, c.a);
        return;
    }
    SDL_SetTextureBlendMode(t, SDL_BLENDMODE_BLEND);
    SDL_SetRenderTarget(ren_, t);
    SDL_SetRenderDrawColor(ren_, 0, 0, 0, 0);
    SDL_RenderClear(ren_);
    stringRGBA(ren_, 0, 0, s.c_str(), c.r, c.g, c.b, c.a);
    SDL_SetRenderTarget(ren_, prev);
    SDL_Rect dst{center ? x - (w * scale) / 2 : x, topY, w * scale, h * scale};
    SDL_RenderCopy(ren_, t, nullptr, &dst);
    SDL_DestroyTexture(t);
}

// Centred pill behind centred text. The pill is sized from the 8x8 bitmap font
// drawText() scales up, so the padding stays proportional at any scale.
void App::drawToast(const std::string& text, int topY, int scale) {
    if (text.empty()) return;
    // Truncate rather than overflow: a webcam's model name can be much wider
    // than the panel at the scale the short labels are sized for.
    std::size_t maxChars = (std::size_t)std::max(4, (int)(viewW_ * 0.92) / (8 * scale));
    std::string s = text;
    if (s.size() > maxChars)
        s = maxChars > 6 ? s.substr(0, maxChars - 3) + "..." : s.substr(0, maxChars);

    int tw = 8 * (int)s.size() * scale;
    int pad = 8 * scale / 2;
    roundedBoxRGBA(ren_, viewW_ / 2 - tw / 2 - pad, topY,
                   viewW_ / 2 + tw / 2 + pad, topY + 8 * scale + pad,
                   6, 0, 0, 0, 120);
    drawText(viewW_ / 2, topY + pad / 2, s, scale, {255, 255, 255, 240}, true);
}

void App::showToast(const std::string& s, Uint32 ms) {
    toast_ = s;
    toastUntil_ = SDL_GetTicks() + ms;
}

// Newest photo/video in the output dir (timestamp names sort chronologically).
static std::string newestMedia(const std::string& dir) {
    std::string best;
    if (DIR* d = ::opendir(dir.c_str())) {
        while (dirent* e = ::readdir(d)) {
            std::string n = e->d_name;
            if (n.find(".video.mp4") != std::string::npos) continue; // mux temp
            if (n.find(".audio.wav") != std::string::npos) continue;
            auto ends = [&](const char* x) {
                std::string s(x);
                return n.size() > s.size() &&
                       n.compare(n.size() - s.size(), s.size(), s) == 0;
            };
            if (ends(".jpg") || ends(".jpeg") || ends(".png") || ends(".mp4") ||
                ends(".avi") || ends(".mov")) {
                if (n > best) best = n; // lexicographic == chronological here
            }
        }
        ::closedir(d);
    }
    return best.empty() ? "" : dir + "/" + best;
}

// Rebuild the little square thumbnail shown on the gallery button.
void App::refreshThumbnail() {
    std::string path = newestMedia(cfg_.outputDir);
    if (path.empty()) {
        if (thumbTex_) { SDL_DestroyTexture(thumbTex_); thumbTex_ = nullptr; }
        thumbPath_.clear();
        return;
    }
    if (path == thumbPath_ && thumbTex_) return;

    cv::Mat img;
    if (Gallery::isVideo(path)) {
        cv::VideoCapture vc(path);
        if (vc.isOpened()) vc.read(img);
    } else {
        img = cv::imread(path, cv::IMREAD_COLOR);
    }
    if (img.empty()) return;

    // Centre-crop to a square, then downscale.
    int side = std::min(img.cols, img.rows);
    cv::Rect roi((img.cols - side) / 2, (img.rows - side) / 2, side, side);
    cv::Mat sq;
    cv::resize(img(roi), sq, cv::Size(128, 128), 0, 0, cv::INTER_AREA);

    if (thumbTex_) { SDL_DestroyTexture(thumbTex_); thumbTex_ = nullptr; }
    thumbTex_ = SDL_CreateTexture(ren_, SDL_PIXELFORMAT_BGR24,
                                  SDL_TEXTUREACCESS_STATIC, sq.cols, sq.rows);
    if (thumbTex_) {
        SDL_UpdateTexture(thumbTex_, nullptr, sq.data, (int)sq.step);
        SDL_SetTextureBlendMode(thumbTex_, SDL_BLENDMODE_BLEND);
    }
    thumbPath_ = path;
}

// Draw the gallery button as the last capture's thumbnail (rounded square),
// falling back to the framed-landscape icon when nothing has been captured.
void App::drawGalleryButton(const Button& b, Uint8 alpha) {
    if (!thumbTex_) {
        Menu::drawButton(ren_, b, alpha);
        return;
    }
    int s = b.r;
    Uint8 bg = (Uint8)((int)alpha * 42 / 100);
    roundedBoxRGBA(ren_, b.cx - s, b.cy - s, b.cx + s, b.cy + s, 7, 18, 18, 24,
                   std::max<Uint8>(1, bg));
    SDL_SetTextureAlphaMod(thumbTex_, alpha);
    SDL_Rect dst{b.cx - s + 3, b.cy - s + 3, 2 * s - 6, 2 * s - 6};
    SDL_RenderCopy(ren_, thumbTex_, nullptr, &dst);
    roundedRectangleRGBA(ren_, b.cx - s, b.cy - s, b.cx + s, b.cy + s, 7,
                         255, 255, 255, (Uint8)((int)alpha * 55 / 100));
}

// Human-readable capture time. Prefers the IMG_/VID_YYYYMMDD_HHMMSS name;
// falls back to the file's mtime.
static std::string captureTime(const std::string& path) {
    std::string base = path.substr(path.find_last_of('/') + 1);
    // Find an 8-4? pattern: _YYYYMMDD_HHMMSS
    for (size_t i = 0; i + 15 < base.size() + 1; ++i) {
        if (base[i] != '_') continue;
        std::string d = base.substr(i + 1, 8), t;
        if (i + 10 <= base.size() && base[i + 9] == '_')
            t = base.substr(i + 10, 6);
        bool digits = d.size() == 8 && t.size() == 6;
        for (char ch : d + t) if (!std::isdigit((unsigned char)ch)) digits = false;
        if (digits)
            return d.substr(0, 4) + "-" + d.substr(4, 2) + "-" + d.substr(6, 2) +
                   "  " + t.substr(0, 2) + ":" + t.substr(2, 2) + ":" +
                   t.substr(4, 2);
    }
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) {
        std::tm tm{};
        localtime_r(&st.st_mtime, &tm);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d  %H:%M:%S", &tm);
        return buf;
    }
    return "";
}

// Battery gauge in the top-right corner: the glyph, the percentage beside it,
// and -- once the pack is nearly flat and off charge -- a pulsing red LOW
// BATTERY warning. Drawn after everything else so it stays readable over a
// bright preview.
void App::drawBatteryBadge() {
    if (!battery_) return;
    const BatteryStatus& b = battery_->status();
    if (!b.valid) return;

    // Roughly the 3.15 V cut-off the HAT shuts down at, expressed as a level.
    constexpr int kLowPercent = 15;
    const bool low = !b.charging && b.percent <= kLowPercent;

    // Pulse the whole badge while low: it catches the eye without stealing the
    // screen from whatever the camera is pointed at.
    Uint8 a = 235;
    if (low) {
        double t = (SDL_GetTicks() % 1400) / 1400.0;
        double tri = (t < 0.5) ? t * 2.0 : (1.0 - t) * 2.0; // 0 -> 1 -> 0
        a = (Uint8)std::lround(120.0 + 135.0 * tri);
    }

    int h = std::max(12, viewH_ / 26);
    int w = (int)std::lround(h * 2.1);
    int m = std::max(8, viewH_ / 48);
    int x = viewW_ - m - w, y = m;
    drawBattery(ren_, x, y, w, h, b.percent, b.charging, a);

    SDL_Color c = low ? SDL_Color{255, 120, 110, a} : SDL_Color{255, 255, 255, a};
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%d%%", b.percent);
    int scale = std::max(1, h / 10);
    int tw = 8 * (int)std::strlen(buf) * scale;
    drawText(x - 8 - tw, y + (h - 8 * scale) / 2, buf, scale, c, false);

    if (!low) return;
    const std::string warn = "LOW BATTERY";
    int wscale = std::max(1, std::min(scale, 2));
    int wx = viewW_ - m - 8 * (int)warn.size() * wscale;
    drawText(wx, y + h + 6, warn, wscale, c, false);
}

// The cell has been under the cut-off, off charge, for a full minute. Leave the
// HAT armed to boot the Pi again once it has charge, then halt cleanly rather
// than letting the pack run down into its own protection cut-off.
void App::powerOffLowBattery() {
    std::cerr << "battery: pack is flat (" << battery_->status().volts
              << " V); shutting down\n";
    // The (D) can be told to boot the Pi again by itself; the (B) has no MCU to
    // ask, so it stays off until someone presses its button.
    if (!battery_->armAutoRestart())
        std::cerr << "battery: this HAT will not power the Pi back up on its "
                     "own; press its button once the pack is charged\n";

    // Tell whoever is watching the screen why it is going dark.
    beginFrame();
    clear();
    int scale = std::max(2, viewH_ / 120);
    drawText(viewW_ / 2, viewH_ / 2 - 4 * scale, "BATTERY EMPTY", scale,
             {255, 120, 110, 255}, true);
    present();
    SDL_Delay(1500);

    // -n so a sudo that would prompt for a password fails fast instead of
    // hanging the app on a machine without passwordless sudo.
    if (std::system("sudo -n poweroff >/dev/null 2>&1") != 0)
        std::system("poweroff >/dev/null 2>&1");
    running_ = false;
}

// Blit a BGR cv::Mat to the screen, preserving aspect ratio (letterboxed).
// Used for decoded gallery/playback frames; the live preview goes through
// blitCamera so it can keep NV12 and zoom on the GPU.
void App::renderMat(const cv::Mat& src, int rotate, const SDL_Rect* crop) {
    clear();
    if (src.empty()) return;

    cv::Mat bgr;
    if (src.type() == CV_8UC3) bgr = src;
    else if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    else if (src.channels() == 1) cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    else bgr = src;

    blitCamera(bgr, PixelFormat::BGR, bgr.cols, bgr.rows, crop, rotate);
}

// Upload a camera frame and blit it letterboxed. For NV12 we hand SDL the raw
// planar buffer and create an NV12 texture, so the GPU (not a CPU core) does
// the YUV->RGB conversion. `src`, when given, is the region to display -- the
// GPU scales it to fill, which is how pinch-zoom stays free of a CPU resize.
void App::blitCamera(const cv::Mat& frame, PixelFormat fmt, int imgW, int imgH,
                     const SDL_Rect* src, int rotate, bool mirror) {
    if (frame.empty() || imgW <= 0 || imgH <= 0) return;

    // Fall back to a CPU convert if the renderer can't sample NV12 textures.
    cv::Mat upload = frame;
    Uint32 sdlFmt = (fmt == PixelFormat::NV12) ? SDL_PIXELFORMAT_NV12
                                               : SDL_PIXELFORMAT_BGR24;
    if (fmt == PixelFormat::NV12 && nv12Unsupported_) {
        cv::cvtColor(frame, bgrScratch_, cv::COLOR_YUV2BGR_NV12);
        upload = bgrScratch_;
        sdlFmt = SDL_PIXELFORMAT_BGR24;
    }

    if (!tex_ || texW_ != imgW || texH_ != imgH || texFmt_ != sdlFmt) {
        if (tex_) SDL_DestroyTexture(tex_);
        tex_ = SDL_CreateTexture(ren_, sdlFmt, SDL_TEXTUREACCESS_STREAMING,
                                 imgW, imgH);
        // First-time NV12 texture rejection: remember it and convert on CPU.
        if (!tex_ && sdlFmt == SDL_PIXELFORMAT_NV12) {
            std::cerr << "display: renderer rejects NV12 textures; converting on "
                         "CPU instead\n";
            nv12Unsupported_ = true;
            cv::cvtColor(frame, bgrScratch_, cv::COLOR_YUV2BGR_NV12);
            upload = bgrScratch_;
            sdlFmt = SDL_PIXELFORMAT_BGR24;
            tex_ = SDL_CreateTexture(ren_, sdlFmt, SDL_TEXTUREACCESS_STREAMING,
                                     imgW, imgH);
        }
        if (!tex_) return;
        SDL_SetTextureBlendMode(tex_, SDL_BLENDMODE_NONE);
        texW_ = imgW;
        texH_ = imgH;
        texFmt_ = sdlFmt;
    }
    SDL_UpdateTexture(tex_, nullptr, upload.data, static_cast<int>(upload.step));

    // Fill the whole view (cover), letting the GPU crop to `src` when zooming.
    // Scaling to the *larger* fit factor makes the image take the panel's exact
    // aspect ratio -- the overflow is clipped by the render target, so there are
    // no letterbox bars. When the camera image is rotated 90/270 its bounding
    // box swaps width/height, so cover against the swapped dimensions and let the
    // GPU spin the frame about its centre (SDL_RenderCopyEx).
    int rot = ((rotate % 360) + 360) % 360;
    bool swap = (rot == 90 || rot == 270);
    double s = swap ? std::max((double)viewW_ / imgH, (double)viewH_ / imgW)
                    : std::max((double)viewW_ / imgW, (double)viewH_ / imgH);
    int dw = (int)(imgW * s), dh = (int)(imgH * s);
    SDL_Rect dst{(viewW_ - dw) / 2, (viewH_ - dh) / 2, dw, dh};
    // SDL flips the texture *before* rotating it, so for a quarter turn the
    // texture's vertical axis is the one that ends up across the screen.
    const SDL_RendererFlip flip =
        !mirror ? SDL_FLIP_NONE : (swap ? SDL_FLIP_VERTICAL : SDL_FLIP_HORIZONTAL);
    if (rot == 0 && flip == SDL_FLIP_NONE) {
        SDL_RenderCopy(ren_, tex_, src, &dst);
    } else {
        SDL_Point center{dw / 2, dh / 2};
        SDL_RenderCopyEx(ren_, tex_, src, &dst, (double)rot, &center, flip);
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

bool App::init(const Config& cfg) {
    cfg_ = cfg;
    if (!initDisplay()) return false;

    // Every camera attached, best first. The first one that actually starts
    // becomes the live preview; the others stay behind the switch button. An
    // entry that won't open is dropped, so the button never offers a camera
    // that isn't there (a board with no Pi camera, an unplugged webcam).
    sources_ = Camera::sources(cfg_);
    while (!cam_ && !sources_.empty()) {
        cam_ = Camera::open(cfg_, sources_.front());
        if (!cam_) sources_.erase(sources_.begin());
    }
    if (!cam_) {
        std::cerr << "startup failed: no camera\n";
        return false;
    }
    std::cout << "camera: " << cam_->description() << " " << cam_->width()
              << "x" << cam_->height() << " @ " << cam_->fps() << "fps\n";
    if (sources_.size() > 1) {
        std::cout << "cameras: " << sources_.size()
                  << " attached; tap the switch button to change\n";
    }

    battery_ = Battery::open(cfg_);
    if (battery_) std::cout << "battery: " << battery_->description() << "\n";

    gallery_ = std::make_unique<Gallery>(cfg_.outputDir);
    if (!cfg_.faceModel.empty()) faceFilter_.setModel(cfg_.faceModel);
    if (!faceFilter_.ready())
        std::cerr << "filters: no MediaPipe face_landmarker.task found; facial "
                     "filters disabled (install the MediaPipe package or pass "
                     "--face-model)\n";
    refreshThumbnail();
    menu_.wake();
    return true;
}

std::string App::timestampName(const char* prefix, const char* ext) const {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "_%Y%m%d_%H%M%S", &tm);
    return cfg_.outputDir + "/" + prefix + buf + ext;
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

void App::pumpEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
            case SDL_QUIT:
                running_ = false;
                break;
            case SDL_KEYDOWN:
                if (mode_ == Mode::Sleep) {
                    wakeFromSleep(); // any key wakes the screen
                } else if (e.key.keysym.sym == SDLK_ESCAPE ||
                           e.key.keysym.sym == SDLK_q) {
                    if (mode_ == Mode::Welcome) running_ = false; // quit the app
                    else if (mode_ == Mode::Camera) goHome();     // back to welcome
                    else mode_ = Mode::Camera;                    // step back to preview
                } else if (mode_ == Mode::Gallery &&
                           (e.key.keysym.sym == SDLK_LEFT ||
                            e.key.keysym.sym == SDLK_RIGHT)) {
                    menu_.wake();
                    dispatch(e.key.keysym.sym == SDLK_LEFT ? Action::Prev
                                                           : Action::Next);
                } else {
                    menu_.wake();
                }
                break;
            case SDL_MOUSEBUTTONDOWN: {
                // Ignore mouse events SDL synthesises from touch (which ==
                // SDL_TOUCH_MOUSEID); the SDL_FINGERDOWN below handles those,
                // otherwise every tap would fire twice.
                if (e.button.which != SDL_TOUCH_MOUSEID) {
                    int vx, vy;
                    physicalToView(e.button.x, e.button.y, vx, vy);
                    onTap(vx, vy);
                }
                break;
            }
            case SDL_FINGERDOWN:
                handleFingerDown(e.tfinger);
                break;
            case SDL_FINGERMOTION:
                handleFingerMotion(e.tfinger);
                break;
            case SDL_FINGERUP:
                handleFingerUp(e.tfinger);
                break;
            default:
                break;
        }
    }
}

void App::mapTouch(float nx, float ny, int& px, int& py) const {
    float x = nx, y = ny;
    switch (cfg_.touchRotate) {          // clockwise, on the unit square
        case 90:  { float t = x; x = 1.f - y; y = t; break; }
        case 180: x = 1.f - x; y = 1.f - y; break;
        case 270: { float t = x; x = y; y = 1.f - t; break; }
        default:  break;
    }
    if (cfg_.touchFlipX) x = 1.f - x;
    if (cfg_.touchFlipY) y = 1.f - y;
    px = std::min(screenW_ - 1, std::max(0, (int)(x * screenW_)));
    py = std::min(screenH_ - 1, std::max(0, (int)(y * screenH_)));
}

double App::fingerSpread() const {
    if (fingers_.size() < 2) return 0.0;
    auto it = fingers_.begin();
    float x1 = it->second.x, y1 = it->second.y;
    ++it;
    float x2 = it->second.x, y2 = it->second.y;
    double dx = (double)(x2 - x1) * screenW_, dy = (double)(y2 - y1) * screenH_;
    return std::sqrt(dx * dx + dy * dy);
}

// A finger whose SDL_FINGERUP never reaches us -- the touch controller or the
// evdev layer drops it now and then -- would otherwise sit in fingers_ for
// good. Every later touch then counts as a second finger: it starts a pinch,
// never becomes a tap, and the menu looks dead until the app is restarted.
// So before tracking a new touch, forget any finger SDL no longer reports as
// down, and any that has sent nothing for longer than a real touch plausibly
// rests motionless (in case SDL lost the lift as well).
void App::pruneStaleFingers(SDL_TouchID touch, Uint32 now) {
    constexpr Uint32 kQuietMs = 2500;
    const int n = SDL_GetNumTouchFingers(touch);
    for (auto it = fingers_.begin(); it != fingers_.end();) {
        bool down = false;
        for (int i = 0; i < n && !down; ++i) {
            const SDL_Finger* sf = SDL_GetTouchFinger(touch, i);
            down = sf && sf->id == it->first;
        }
        if (!down || now - it->second.lastMs > kQuietMs) it = fingers_.erase(it);
        else ++it;
    }
    if (fingers_.size() < 2) pinching_ = false;
}

void App::handleFingerDown(const SDL_TouchFingerEvent& f) {
    pruneStaleFingers(f.touchId, f.timestamp);
    menu_.wake();
    fingers_[f.fingerId] = {f.x, f.y, f.timestamp};
    if (fingers_.size() == 1) {
        // Possible tap; confirmed on finger-up if it stays put and no 2nd finger.
        tapCandidate_ = true;
        tapFinger_ = f.fingerId;
        tapStartX_ = f.x;
        tapStartY_ = f.y;
        tapStartMs_ = f.timestamp;
    } else if (fingers_.size() == 2) {
        // A second finger starts a pinch and cancels the pending tap.
        tapCandidate_ = false;
        pinching_ = true;
        pinchStartDist_ = fingerSpread();
        zoomLabelUntil_ = SDL_GetTicks() + 1200;
        if (mode_ == Mode::Gallery) {
            // The photo, not the camera behind it. Remember which point of
            // the image is under the fingers, so the zoom grows out of it.
            pinchStartZoom_ = galleryZoom_;
            double mx, my;
            if (fingerMidView(mx, my))
                galleryViewToImage(mx, my, pinchAnchorX_, pinchAnchorY_);
        } else if (cam_) {
            pinchStartZoom_ = cam_->zoom();
        }
    }
}

void App::handleFingerMotion(const SDL_TouchFingerEvent& f) {
    auto it = fingers_.find(f.fingerId);
    float lastX = f.x, lastY = f.y;
    if (it != fingers_.end()) {
        lastX = it->second.x;
        lastY = it->second.y;
        it->second = {f.x, f.y, f.timestamp};
    }

    const bool photo = mode_ == Mode::Gallery && galleryZoomable();
    if (pinching_ && fingers_.size() >= 2 && pinchStartDist_ > 1.0) {
        double z = pinchStartZoom_ * (fingerSpread() / pinchStartDist_);
        if (photo) {
            // Zoom about the fingers: keep the image point that was under
            // them at the start under their midpoint now, which also pans
            // the photo along with a two-finger drag.
            galleryZoom_ = std::min(kGalleryMaxZoom, std::max(1.0, z));
            double mx, my;
            const double s = galleryCoverScale() * galleryZoom_;
            if (fingerMidView(mx, my) && s > 0.0) {
                galleryPanX_ = pinchAnchorX_ - (mx - viewW_ * 0.5) / (s * galleryMat_.cols);
                galleryPanY_ = pinchAnchorY_ - (my - viewH_ * 0.5) / (s * galleryMat_.rows);
            }
            galleryCrop(); // clamp the pan
        } else if (mode_ != Mode::Gallery && cam_) {
            cam_->setZoom(z);
        }
        zoomLabelUntil_ = SDL_GetTicks() + 1200;
    } else if (photo && galleryZoom_ > 1.001 && fingers_.size() == 1 &&
               it != fingers_.end()) {
        // One finger drags a zoomed photo around, the way it would on a phone.
        double x0, y0, x1, y1;
        fingerToView(lastX, lastY, x0, y0);
        fingerToView(f.x, f.y, x1, y1);
        const double s = galleryCoverScale() * galleryZoom_;
        if (s > 0.0) {
            galleryPanX_ -= (x1 - x0) / (s * galleryMat_.cols);
            galleryPanY_ -= (y1 - y0) / (s * galleryMat_.rows);
            galleryCrop();
        }
    }
    if (tapCandidate_ && f.fingerId == tapFinger_) {
        float dx = f.x - tapStartX_, dy = f.y - tapStartY_;
        if (dx * dx + dy * dy > 0.0009f) tapCandidate_ = false; // ~3% drag
    }
}

void App::handleFingerUp(const SDL_TouchFingerEvent& f) {
    fingers_.erase(f.fingerId);
    if (fingers_.size() < 2) pinching_ = false;

    // Timed on the events' own timestamps: a slow frame (a camera switch, a
    // heavy filter) can hold the up event in the queue well past the limit, and
    // timing it on arrival used to throw away a perfectly quick tap.
    if (tapCandidate_ && f.fingerId == tapFinger_ && fingers_.empty()) {
        if (f.timestamp - tapStartMs_ < 700) {
            int px, py, vx, vy;
            mapTouch(f.x, f.y, px, py);
            physicalToView(px, py, vx, vy);
            onTap(vx, vy);
        }
    }
    tapCandidate_ = false;
}

// Normalised touch coordinates to the logical view, through the touch panel's
// own mapping and the display rotation -- the same path a tap takes.
void App::fingerToView(float nx, float ny, double& vx, double& vy) const {
    int px, py, ix, iy;
    mapTouch(nx, ny, px, py);
    physicalToView(px, py, ix, iy);
    vx = ix;
    vy = iy;
}

bool App::fingerMidView(double& vx, double& vy) const {
    if (fingers_.size() < 2) return false;
    auto it = fingers_.begin();
    double ax, ay, bx, by;
    fingerToView(it->second.x, it->second.y, ax, ay);
    ++it;
    fingerToView(it->second.x, it->second.y, bx, by);
    vx = 0.5 * (ax + bx);
    vy = 0.5 * (ay + by);
    return true;
}

void App::resetGalleryZoom() {
    galleryZoom_ = 1.0;
    galleryPanX_ = galleryPanY_ = 0.5;
    pinching_ = false; // a pinch in progress belonged to the previous photo
}

bool App::galleryZoomable() const {
    return gallery_ && !gallery_->empty() && !gallery_->currentIsVideo() &&
           !galleryMat_.empty();
}

// renderMat fills the screen with the image ("cover"), so at 1x one image
// pixel is this many screen pixels, and at zoom z, z times that.
double App::galleryCoverScale() const {
    if (galleryMat_.empty()) return 0.0;
    return std::max((double)viewW_ / galleryMat_.cols,
                    (double)viewH_ / galleryMat_.rows);
}

void App::galleryViewToImage(double vx, double vy, double& u, double& v) const {
    const double s = galleryCoverScale() * galleryZoom_;
    if (s <= 0.0) { u = v = 0.5; return; }
    u = galleryPanX_ + (vx - viewW_ * 0.5) / (s * galleryMat_.cols);
    v = galleryPanY_ + (vy - viewH_ * 0.5) / (s * galleryMat_.rows);
}

SDL_Rect App::galleryCrop() {
    const int iw = galleryMat_.cols, ih = galleryMat_.rows;
    if (iw <= 0 || ih <= 0) return SDL_Rect{0, 0, 0, 0};
    const double z = std::max(1.0, galleryZoom_);
    // A crop of 1/z of the image each way, kept on the image. renderMat
    // centres it on the screen, so the screen centre is the pan point exactly
    // and the pinch's anchor maths holds right up to the edges. (Where the
    // screen's shape differs from the photo's, cover trims a sliver off one
    // axis that the pan cannot reach -- the same sliver trimmed at 1x.)
    const double half = 0.5 / z;
    galleryPanX_ = std::min(1.0 - half, std::max(half, galleryPanX_));
    galleryPanY_ = std::min(1.0 - half, std::max(half, galleryPanY_));
    const int cw = std::max(1, (int)std::lround(iw / z));
    const int ch = std::max(1, (int)std::lround(ih / z));
    const int cx = std::min(iw - cw, std::max(0, (int)std::lround(galleryPanX_ * iw) - cw / 2));
    const int cy = std::min(ih - ch, std::max(0, (int)std::lround(galleryPanY_ * ih) - ch / 2));
    return SDL_Rect{cx, cy, cw, ch};
}

void App::onTap(int x, int y) {
    if (mode_ == Mode::Sleep) {
        // Wake only on a double-tap, so a stray touch keeps the screen asleep.
        Uint32 now = SDL_GetTicks();
        if (lastSleepTapMs_ && now - lastSleepTapMs_ < 600) {
            wakeFromSleep();
            lastSleepTapMs_ = 0;
        } else {
            lastSleepTapMs_ = now;
        }
        return;
    }

    if (mode_ == Mode::Welcome) {
        // Welcome buttons are always shown, so a tap acts immediately.
        menu_.wake();
        auto btns = buttonsFor(Mode::Welcome);
        dispatch(Menu::hitTest(btns, x, y));
        return;
    }

    if (mode_ == Mode::ConfirmDelete) {
        auto btns = buttonsFor(mode_);
        Action a = Menu::hitTest(btns, x, y);
        dispatch(a == Action::ConfirmYes ? Action::ConfirmYes : Action::ConfirmNo);
        return;
    }

    // In Camera/Gallery, a tap while the menu is hidden only wakes it.
    bool wasAwake = menu_.awake();
    menu_.wake();
    if (!wasAwake) return;

    auto btns = buttonsFor(mode_);
    dispatch(Menu::hitTest(btns, x, y));
}

std::vector<Button> App::buttonsFor(Mode m) const {
    bool hasVideo = gallery_ && !gallery_->empty() && gallery_->currentIsVideo();
    return menu_.layout(m, viewW_, viewH_, hasVideo, sources_.size() > 1);
}

void App::dispatch(Action a) {
    switch (a) {
        case Action::Shutter:     capturePhoto(); break;
        case Action::ZoomIn:      cam_->zoomIn(); break;
        case Action::ZoomOut:     cam_->zoomOut(); break;
        case Action::OpenGallery:
            gallery_->refresh(); gallery_->selectNewest(); refreshThumbnail();
            resetGalleryZoom();
            mode_ = Mode::Gallery;
            break;
        case Action::Back:        mode_ = Mode::Camera; break;
        // The strip reads like a timeline: the left arrow goes back in time.
        case Action::Prev:        gallery_->older(); resetGalleryZoom(); break;
        case Action::Next:        gallery_->newer(); resetGalleryZoom(); break;
        case Action::Play:        playCurrentVideo(); break;
        case Action::Delete:
            if (gallery_ && !gallery_->empty()) mode_ = Mode::ConfirmDelete;
            break;
        case Action::ConfirmYes:
            gallery_->deleteCurrent();
            refreshThumbnail();
            resetGalleryZoom();
            mode_ = gallery_->empty() ? Mode::Camera : Mode::Gallery;
            break;
        case Action::ConfirmNo:
            mode_ = Mode::Gallery;
            break;
        case Action::CycleFilter:
            filter_ = nextFilter(filter_);
            showToast(filterName(filter_), 1500);
            break;
        case Action::SwitchCamera:
            switchCamera();
            break;
        case Action::StartCamera:
            refreshSources(); // a webcam plugged in since startup gets the button
            // Normally already open again since the wake; this is the retry
            // for a camera that would not start then.
            if (!reopenCamera()) break;
            mode_ = Mode::Camera;
            menu_.wake();
            break;
        case Action::Sleep:
            enterSleep();
            break;
        case Action::Home:
            goHome();
            break;
        case Action::Quit:
            running_ = false;
            break;
        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

void App::capturePhoto() {
    if (lastNative_.empty()) return;
    // Guard against the output dir having gone missing since startup (e.g. an
    // unmounted SD/USB path) so the write below can actually land.
    ensureDir(cfg_.outputDir);
    // Materialise a BGR frame on demand (the preview kept it in the camera's
    // native format). Reshape at full resolution, then zoom -- the same order as
    // the preview -- so the saved photo matches what's on screen.
    cv::Mat shot = cam_->nativeToBGR(lastNative_);
    if (shot.empty()) return;
    shot = rotatedBGR(shot, cfg_.cameraRotate); // to display orientation first, so
                                                // the face detector sees an upright
                                                // face; matches the preview
    faceFilter_.apply(shot, filter_, filterPhase_);
    cam_->cropZoom(shot);
    // Mirrored last, as the preview is (see renderCamera), so the photo is
    // exactly what was on screen -- filter lighting and all.
    if (mirrorLive()) cv::flip(shot, shot, 1);
    std::string path = timestampName("IMG", ".jpg");
    bool ok = false;
    try {
        ok = cv::imwrite(path, shot);
    } catch (const cv::Exception& e) {
        std::cerr << "imwrite threw for " << path << ": " << e.what() << "\n";
    }
    if (!ok) {
        std::cerr << "failed to save " << path << "\n";
        return;
    }
    std::cout << "saved " << path << "\n";
    flashStart_ = SDL_GetTicks(); // shutter flash animation
    refreshThumbnail();           // update the gallery-button preview
}

// Open `s`, giving it one more chance after a short pause: the first open of a
// device that was only just released (by us, a moment ago) often fails.
std::unique_ptr<Camera> App::openWithRetry(const CameraSource& s) {
    if (auto cam = Camera::open(cfg_, s)) return cam;
    SDL_Delay(400);
    return Camera::open(cfg_, s);
}

// Re-read the attached webcams. The Pi camera is kept only if startup proved
// it is there, because sources() lists it whether or not a sensor is fitted.
// The live camera always stays, so the rotation can find its place.
void App::refreshSources() {
    const bool havePi = std::any_of(sources_.begin(), sources_.end(),
                                    [](const CameraSource& s) {
                                        return s.kind == CameraKind::PiCam;
                                    });
    std::vector<CameraSource> fresh;
    for (const CameraSource& s : Camera::sources(cfg_))
        if (s.kind != CameraKind::PiCam || havePi) fresh.push_back(s);
    if (cam_ && std::find(fresh.begin(), fresh.end(), cam_->source()) == fresh.end())
        fresh.insert(fresh.begin(), cam_->source());
    sources_ = std::move(fresh);
}

// Move to the next attached camera. The live device is released *before* the
// next one is opened: two USB cameras on one controller can easily want more
// bandwidth than the bus has, and holding both open would make the second fail
// to start for a reason that has nothing to do with the second.
//
// A camera that won't start here is skipped, not forgotten. Releasing a device
// is not instant -- libcamera and the UVC driver both take a moment to let go
// -- so a camera that was live a second ago routinely refuses its first
// reopen. Dropping it on that used to take the switch button away for the rest
// of the session as soon as the list fell to one camera. Webcams are
// re-enumerated instead, so one that really was unplugged leaves the list (and
// one plugged in since startup joins it).
void App::switchCamera() {
    if (!cam_) return;
    refreshSources();
    if (sources_.size() < 2) return;

    const CameraSource current = cam_->source();
    const double zoom = cam_->zoom(); // the user's framing, not the device's
    auto it = std::find(sources_.begin(), sources_.end(), current);
    const std::size_t at = (std::size_t)(it - sources_.begin()) % sources_.size();

    cam_.reset();
    std::unique_ptr<Camera> next;
    for (std::size_t step = 1; step < sources_.size() && !next; ++step) {
        const CameraSource& s = sources_[(at + step) % sources_.size()];
        next = openWithRetry(s);
        if (!next) std::cerr << "camera: " << s.label << " would not start\n";
    }

    if (!next) {
        // Nothing else worked: go back to the camera that was already running.
        next = openWithRetry(current);
        if (!next) {
            std::cerr << "camera: lost " << current.label
                      << " and no other camera could be opened\n";
            running_ = false;
            return;
        }
    }
    cam_ = std::move(next);

    // The new source has its own size and pixel format, so the buffers held for
    // the old one describe nothing now; blitCamera() rebuilds its texture from
    // the next frame it is handed.
    lastNative_.release();
    filteredNative_.release();
    cam_->setZoom(zoom);

    showToast(cam_->source().label, 1800);
    menu_.wake();
    std::cout << "camera: " << cam_->description() << " " << cam_->width()
              << "x" << cam_->height() << " @ " << cam_->fps() << "fps\n";
}

// Blocking playback of the selected video: renders frames at the source fps and
// returns to the gallery on end, tap or key.
void App::playCurrentVideo() {
    if (!gallery_ || gallery_->empty() || !gallery_->currentIsVideo()) return;
    cv::VideoCapture vc(gallery_->current());
    if (!vc.isOpened()) return;

    mode_ = Mode::Playback;
    double fps = vc.get(cv::CAP_PROP_FPS);
    Uint32 frameMs = (Uint32)(fps > 1.0 ? 1000.0 / fps : 33.0);

    cv::Mat frame;
    bool stop = false;
    while (running_ && !stop && vc.read(frame) && !frame.empty()) {
        Uint32 t0 = SDL_GetTicks();
        // Playback owns the loop for the length of the clip, so keep the gauge
        // ticking here too -- otherwise a long video leaves the low-battery
        // timer frozen at whatever it read before playback started.
        if (battery_) battery_->poll();
        beginFrame();
        renderMat(frame);
        present();

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) { running_ = false; stop = true; }
            else if (e.type == SDL_KEYDOWN || e.type == SDL_MOUSEBUTTONDOWN ||
                     e.type == SDL_FINGERDOWN) stop = true;
        }
        Uint32 dt = SDL_GetTicks() - t0;
        if (dt < frameMs) SDL_Delay(frameMs - dt);
    }
    mode_ = Mode::Gallery;
    menu_.wake();
}

// Leave the live camera and return to the welcome screen.
void App::goHome() {
    mode_ = Mode::Welcome;
    menu_.wake();
}

// Sleep. A black screen alone saves next to nothing: the backlight is most of
// an LCD's draw and stays lit behind black pixels, and the camera keeps
// streaming -- sensor, ISP and the capture pipeline all running -- for a
// preview nobody is looking at. So the camera is closed, the backlight and the
// display pipeline are switched off (DisplayPower), and the main loop blocks
// waiting for a touch instead of spinning (renderSleep). The touch controller
// is a separate I2C device and stays awake, so the wake double-tap (see onTap)
// still works, as does any key.
void App::enterSleep() {
    mode_ = Mode::Sleep;
    lastSleepTapMs_ = 0;
    // Flush a black frame first, so nothing lingers on a panel that cannot be
    // switched off and nothing flashes up on one that comes back.
    beginFrame();
    clear();
    present();

    if (cam_) {
        sleptCamera_ = cam_->source();
        sleptZoom_ = cam_->zoom();
        // Frames may point into the capture pipeline's buffers; let go of
        // them before the pipeline goes.
        lastNative_.release();
        filteredNative_.release();
        cam_.reset();
        std::cout << "camera: closed for sleep\n";
    }

    // Not in a desktop session: there the "display" is a window, and the
    // backlight is the laptop's.
    const bool desktop = cfg_.windowed || std::getenv("DISPLAY") ||
                         std::getenv("WAYLAND_DISPLAY");
    const std::string off = desktop ? std::string() : displayPower_.sleep(win_);
    std::cout << (off.empty() ? std::string("display: screen blanked (sleep); "
                                            "nothing could be switched off\n")
                              : "display: switched off " + off + " (sleep)\n");
}

bool App::reopenCamera() {
    if (cam_) return true;
    std::vector<CameraSource> order{sleptCamera_};
    for (const CameraSource& s : sources_)
        if (!(s == sleptCamera_)) order.push_back(s);
    for (const CameraSource& s : order) {
        if (s.index < 0 && s.kind == CameraKind::Webcam) continue; // never set
        cam_ = openWithRetry(s);
        if (cam_) break;
    }
    if (!cam_) {
        std::cerr << "camera: none would start after sleep\n";
        return false;
    }
    if (cam_->source() == sleptCamera_) cam_->setZoom(sleptZoom_);
    std::cout << "camera: " << cam_->description() << " reopened\n";
    return true;
}

// Restore the display, bring the camera back and go to the welcome screen.
// The camera is reopened now rather than on Start, so Start is instant; the
// second or so it takes passes while the welcome screen is coming up.
void App::wakeFromSleep() {
    displayPower_.wake(win_);
    mode_ = Mode::Welcome;
    menu_.wake();
    renderWelcome(); // a picture up before the camera's start-up pause
    reopenCamera();
    std::cout << "display: woke from sleep\n";
}

// ---------------------------------------------------------------------------
// Per-mode rendering
// ---------------------------------------------------------------------------

// Start screen: a camera built from Lego bricks, a title, and two big controls
// -- Start Camera and Sleep -- with text labels.
void App::renderWelcome() {
    beginFrame();

    // Background: a soft vertical gradient from deep blue to near-black.
    for (int y = 0; y < viewH_; ++y) {
        double f = viewH_ > 1 ? (double)y / (viewH_ - 1) : 0.0;
        Uint8 r = (Uint8)(18 * (1 - f) + 6 * f);
        Uint8 g = (Uint8)(22 * (1 - f) + 7 * f);
        Uint8 b = (Uint8)(44 * (1 - f) + 14 * f);
        SDL_SetRenderDrawColor(ren_, r, g, b, 255);
        SDL_RenderDrawLine(ren_, 0, y, viewW_, y);
    }

    // Title, scaled to fit ~86% of the width.
    const std::string title = "OPEN LEGO CAMERA";
    int fitW = (int)(viewW_ * 0.86 / (8 * (int)title.size()));
    int tscale = std::max(2, std::min({fitW, viewH_ / 60, 6}));
    int titleY = std::max(14, viewH_ / 14);
    drawText(viewW_ / 2, titleY, title, tscale, {255, 214, 40, 245}, true);

    // The Lego-brick camera, centred in the upper-middle of the screen.
    double unit = std::min(viewW_ / 13.0, viewH_ / 12.0);
    drawLegoCamera(ren_, viewW_ / 2, (int)(viewH_ * 0.42), unit, 255);

    // Two always-visible controls with labels beneath them.
    auto btns = buttonsFor(Mode::Welcome);
    for (const auto& b : btns) Menu::drawButton(ren_, b, 255);
    int lscale = std::max(2, std::min(viewH_ / 220, 3));
    for (const auto& b : btns) {
        const char* label = (b.action == Action::StartCamera) ? "START" : "SLEEP";
        drawText(b.cx, b.cy + b.r + 12, label, lscale, {255, 255, 255, 235}, true);
    }

    drawBatteryBadge();

    // Footer hint about waking from sleep.
    const std::string hint = "DOUBLE-TAP SCREEN TO WAKE FROM SLEEP";
    int hFit = (int)(viewW_ * 0.9 / (8 * (int)hint.size()));
    int hscale = std::max(1, std::min(hFit, 2));
    drawText(viewW_ / 2, viewH_ - 16 - 8 * hscale, hint, hscale,
             {150, 160, 180, 200}, true);

    present();
}

// Asleep: nothing to draw, so block until there is input rather than polling
// for it. The timeout only bounds how long the main loop goes without seeing
// the battery -- it still has to shut the Pi down cleanly if the cell runs
// flat overnight -- and once a second is plenty for that. Where nothing could
// be switched off, the black frame is re-presented at the same rate in case
// anything else draws over it.
void App::renderSleep() {
    if (!displayPower_.dark()) {
        beginFrame();
        clear();
        present();
    }
    SDL_WaitEventTimeout(nullptr, 1000);
}

void App::renderCamera() {
    cv::Mat frame;
    if (cam_->read(frame)) lastNative_ = frame; // camera-native, no zoom applied

    const bool filtering = (filter_ != Filter::None);
    // An NV12 frame can be filtered without a full-frame CPU convert -- only the
    // face region is reshaped and re-encoded, the GPU converts and zooms the
    // rest (see renderFilteredNV12). A BGR webcam or a renderer without NV12
    // textures falls back to converting the full frame.
    //
    // MediaPipe's face detector only finds roughly upright faces, so filtering
    // must run on the frame in *display* orientation. When --camera-rotate spins
    // the image (e.g. an upside-down camera corrected with 180), the NV12 fast
    // path -- which detects and reshapes on the un-rotated native buffer and
    // rotates only at blit -- would look for an upright face in a rotated frame
    // and find none, so the filter silently did nothing. Restrict the fast path
    // to the unrotated case and route rotated filtering through the BGR path
    // below, which rotates to display orientation *before* detecting and
    // reshaping.
    const bool nv12FilterPath = filtering &&
                                cam_->format() == PixelFormat::NV12 &&
                                !nv12Unsupported_ && !lastNative_.empty() &&
                                cfg_.cameraRotate == 0;

    // Mirroring is always the last step, after the filter has drawn: the
    // face tracking then sees the camera's own image whichever way it is
    // shown, and the NV12 paths can leave the flip to the GPU.
    const bool mirror = mirrorLive();

    beginFrame();
    if (filtering && !nv12FilterPath) {
        // Full-resolution BGR: convert, rotate to display orientation, reshape
        // the face, then zoom. Reshaping after the rotation lets the upright-only
        // detector find the face; running it before the zoom crop (and at full
        // resolution) matches the NV12 preview path so a photo carries exactly the
        // expression shown on screen.
        cv::Mat bgr = cam_->nativeToBGR(lastNative_);
        bgr = rotatedBGR(bgr, cfg_.cameraRotate);
        faceFilter_.apply(bgr, filter_, filterPhase_);
        cam_->cropZoom(bgr);
        if (mirror) cv::flip(bgr, bgr, 1);
        renderMat(bgr, 0); // already in display orientation
    } else if (nv12FilterPath) {
        renderFilteredNV12();
    } else if (!lastNative_.empty()) {
        // Pure preview: no CPU colour-convert or resize at all.
        clear();
        cv::Rect zr = cam_->zoomSrcRect(cam_->width(), cam_->height());
        SDL_Rect z{zr.x, zr.y, zr.width, zr.height};
        blitCamera(lastNative_, cam_->format(), cam_->width(), cam_->height(),
                   cam_->zoomed() ? &z : nullptr, cfg_.cameraRotate, mirror);
    } else {
        clear();
    }

    if (filter_ != Filter::None) filterPhase_ += 1.0;

    if (menu_.awake()) {
        Uint8 a = menu_.alpha();
        auto btns = buttonsFor(Mode::Camera);
        for (const auto& b : btns) {
            if (b.action == Action::OpenGallery)
                drawGalleryButton(b, a); // last-shot thumbnail
            else
                Menu::drawButton(ren_, b, a);
        }
    }

    // Magnification factor while/just after pinching (or whenever zoomed in).
    Uint32 now = SDL_GetTicks();
    double z = cam_->zoom();
    if (z > 1.001 || now < zoomLabelUntil_) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.1fx", z);
        drawToast(buf, 12, std::max(2, viewH_ / 160));
    }

    // Filter or camera name, shown briefly after tapping the button that
    // changed it.
    if (now < toastUntil_) drawToast(toast_, viewH_ / 6, std::max(2, viewH_ / 200));

    // Shutter flash: a quick white wash that fades out after a capture.
    if (flashStart_) {
        Uint32 dt = now - flashStart_;
        const Uint32 dur = 320;
        if (dt < dur) {
            Uint8 a = (Uint8)(210.0 * (1.0 - (double)dt / dur));
            boxRGBA(ren_, 0, 0, viewW_, viewH_, 255, 255, 255, a);
        } else {
            flashStart_ = 0;
        }
    }

    drawBatteryBadge();
    present();
}

// Filtered preview that keeps the frame in NV12. Detection gets a small colour
// image built by downscaling the Y and UV planes separately (never a
// full-frame convert), and only the face region is converted to BGR, reshaped,
// and re-encoded back into a private NV12 copy. The GPU then does the YUV->RGB
// conversion and the zoom crop for the whole frame, exactly as on the
// unfiltered fast path -- so turning a filter on still costs work proportional
// to the face, not to the frame.
void App::renderFilteredNV12() {
    const int W = cam_->width(), H = cam_->height();
    // The detection image is pre-scaled, so the frame size has to be passed
    // explicitly -- landmarks are normalized and get multiplied up into frame
    // coordinates.
    faceFilter_.updateDetection(
        Camera::nv12ToBGRScaled(lastNative_, FaceFilter::detectionWidth()),
        cv::Size(W, H));
    cv::Rect region = faceFilter_.dirtyRegion(filter_, W, H, filterPhase_);

    clear();
    cv::Rect zr = cam_->zoomSrcRect(W, H);
    SDL_Rect z{zr.x, zr.y, zr.width, zr.height};
    const SDL_Rect* zp = cam_->zoomed() ? &z : nullptr;

    if (region.area() == 0) {
        // No face in view: nothing to reshape, so stay on the pure fast path.
        blitCamera(lastNative_, PixelFormat::NV12, W, H, zp, cfg_.cameraRotate,
                   mirrorLive());
        return;
    }

    // Reshape the region on a private copy so the shared capture buffer (used by
    // stills) is never mutated, then upload the whole NV12 once.
    lastNative_.copyTo(filteredNative_);
    cv::Mat roi = Camera::nv12CropToBGR(filteredNative_, region);
    faceFilter_.applyRegion(roi, region.tl(), filter_, filterPhase_);
    Camera::bgrIntoNV12(roi, filteredNative_, region.tl());
    blitCamera(filteredNative_, PixelFormat::NV12, W, H, zp, cfg_.cameraRotate,
               mirrorLive());
}

bool App::mirrorLive() const {
    if (!cam_) return false;
    switch (cfg_.mirror) {
        case Mirror::None:   return false;
        case Mirror::All:    return true;
        case Mirror::PiCam:  return cam_->source().kind == CameraKind::PiCam;
        case Mirror::Webcam: return cam_->source().kind == CameraKind::Webcam;
    }
    return false;
}

void App::ensureGalleryImage() {
    if (gallery_->empty()) { galleryMat_.release(); galleryShown_.clear(); return; }
    const std::string& path = gallery_->current();
    if (path == galleryShown_ && !galleryMat_.empty()) return;

    if (Gallery::isVideo(path)) {
        cv::VideoCapture vc(path);
        cv::Mat first;
        if (vc.isOpened()) vc.read(first);
        galleryMat_ = first;
    } else {
        galleryMat_ = cv::imread(path, cv::IMREAD_COLOR);
    }
    galleryShown_ = path;
}

void App::renderGallery() {
    ensureGalleryImage();

    beginFrame();
    if (galleryMat_.empty()) {
        clear(); // nothing captured yet: black with just the Back control
    } else {
        const SDL_Rect crop = galleryCrop();
        renderMat(galleryMat_, 0, galleryZoom_ > 1.001 ? &crop : nullptr);
        // A centred play glyph hints that the current item is a video.
        if (gallery_->currentIsVideo()) {
            int r = std::max(30, viewH_ / 10);
            filledCircleRGBA(ren_, viewW_ / 2, viewH_ / 2, r, 0, 0, 0, 90);
            drawIcon(ren_, Action::Play, viewW_ / 2, viewH_ / 2,
                     (int)(r * 0.62), 220);
        }
    }

    // Capture date/time, translucent, across the top.
    if (!gallery_->empty()) {
        std::string ts = captureTime(gallery_->current());
        if (!ts.empty()) {
            // Scale for readability but shrink so it fits the width in portrait.
            int fitW = (int)(viewW_ * 0.92 / (8 * ts.size()));
            int scale = std::max(2, std::min({viewH_ / 150, fitW, 4}));
            int barH = 8 * scale + 20;
            boxRGBA(ren_, 0, 0, viewW_, barH, 0, 0, 0, 105);
            drawText(viewW_ / 2, (barH - 8 * scale) / 2, ts, scale,
                     {255, 255, 255, 220}, true);
        }
    }

    // Magnification while/just after pinching, or whenever zoomed in.
    if (galleryZoomable() &&
        (galleryZoom_ > 1.001 || SDL_GetTicks() < zoomLabelUntil_)) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.1fx", galleryZoom_);
        drawToast(buf, viewH_ / 8, std::max(2, viewH_ / 160));
    }

    Uint8 a = menu_.awake() ? menu_.alpha() : (Uint8)0;
    if (a > 0) {
        // Oldest is on the left, newest on the right: an arrow with nothing
        // further that way is drawn faint (it stays in place so the row never
        // shifts under a finger tapping through the photos).
        auto btns = buttonsFor(Mode::Gallery);
        for (const auto& b : btns) {
            bool dead = (b.action == Action::Prev && gallery_->atOldest()) ||
                        (b.action == Action::Next && gallery_->atNewest());
            Menu::drawButton(ren_, b, dead ? (Uint8)(a / 3) : a);
        }
    }

    drawBatteryBadge();
    present();
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

int App::run() {
    while (running_) {
        pumpEvents();
        if (!running_) break;

        if (battery_) {
            battery_->poll(); // only touches the bus every couple of seconds
            if (cfg_.batteryShutdown && battery_->criticallyLow()) {
                powerOffLowBattery();
                break;
            }
        }

        switch (mode_) {
            case Mode::Welcome:
                renderWelcome();
                break;
            case Mode::Sleep:
                renderSleep();
                break;
            case Mode::Camera:
                renderCamera();
                break;
            case Mode::Gallery:
                renderGallery();
                break;
            case Mode::ConfirmDelete: {
                // Dim the shown item, then two always-on confirm buttons.
                ensureGalleryImage();
                beginFrame();
                if (!galleryMat_.empty()) renderMat(galleryMat_);
                else clear();
                boxRGBA(ren_, 0, 0, viewW_, viewH_, 0, 0, 0, 120);
                auto btns = buttonsFor(Mode::ConfirmDelete);
                for (const auto& b : btns) Menu::drawButton(ren_, b, 255);
                present();
                break;
            }
            case Mode::Playback:
                // Playback runs its own loop in playCurrentVideo(); nothing here.
                break;
        }
    }
    return 0;
}

} // namespace olc
