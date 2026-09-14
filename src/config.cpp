#include "config.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

namespace olc {

std::string defaultOutputDir() {
    const char* home = std::getenv("HOME");
    std::string base = home ? std::string(home) : std::string(".");
    return base + "/Pictures/open-lego-camera";
}

// Create `path` and every missing parent (like `mkdir -p`). Returns true when
// the full directory exists afterwards. A plain ::mkdir only creates the leaf,
// so on a headless Raspberry Pi OS Lite install -- which is what this app
// wants, since it drives KMS/DRM directly -- ~/Pictures doesn't exist yet and
// it fails with ENOENT, silently losing captures. Desktop images ship
// ~/Pictures via xdg-user-dirs, which is why the single-level mkdir happened
// to work there.
bool ensureDir(const std::string& path) {
    if (path.empty()) return false;
    std::string partial;
    partial.reserve(path.size());
    for (std::size_t i = 0; i < path.size(); ++i) {
        partial += path[i];
        const bool atSep = (path[i] == '/');
        const bool atEnd = (i + 1 == path.size());
        if ((atSep && partial.size() > 1) || atEnd) {
            std::string dir = atSep ? partial.substr(0, partial.size() - 1) : partial;
            if (dir.empty()) continue; // leading '/'
            if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
                std::cerr << "could not create directory " << dir << ": "
                          << std::strerror(errno) << "\n";
                return false;
            }
        }
    }
    return true;
}

static void printUsage(const char* prog) {
    std::cout <<
        "open-lego-camera - icon-only camera for the Raspberry Pi 5\n\n"
        "Usage: " << prog << " [options]\n\n"
        "  --camera auto|picam|webcam   camera source (default: auto)\n"
        "  --output-dir DIR             where captures are saved\n"
        "                               (default: ~/Pictures/open-lego-camera)\n"
        "  --webcam-index N             force /dev/videoN for a USB webcam\n"
        "  --picam-name NAME            libcamera camera id to select (see\n"
        "                               `rpicam-hello --list-cameras`)\n"
        "  --size WxH                   requested preview size (default: 1280x720)\n"
        "  --rotate 0|90|180|270        rotate the whole UI (preview + buttons)\n"
        "                               to match a rotated panel\n"
        "  --camera-rotate 0|90|180|270 rotate only the camera image (preview and\n"
        "                               captures), leaving the buttons in place\n"
        "  --touch-rotate 0|90|180|270  extra touch rotation if the touch panel\n"
        "                               is misaligned from the display\n"
        "                               (e.g. Pimoroni HyperPixel)\n"
        "  --touch-flip-x               mirror touch horizontally\n"
        "  --touch-flip-y               mirror touch vertically\n"
        "  --driver NAME                force SDL video driver (kmsdrm, fbcon, x11)\n"
        "  --windowed                   run in a window instead of fullscreen\n"
        "  --face-model PATH            MediaPipe face_landmarker.task bundle\n"
        "                               driving the facial filters (default: the\n"
        "                               packaged model under /opt/mediapipe or\n"
        "                               /usr/share/mediapipe/models)\n"
        "  --no-battery                 skip the Waveshare UPS HAT (D) battery\n"
        "                               gauge (it is auto-detected otherwise)\n"
        "  --battery-bus N              I2C bus the UPS HAT is on (default: 1)\n"
        "  --battery-shutdown           power the Pi off when the cell reaches\n"
        "                               the 3.15 V cut-off (off by default)\n"
        "  --help                       show this help\n";
}

// Parse "WxH" into width/height. Returns false on malformed input.
static bool parseSize(const std::string& s, int& w, int& h) {
    auto x = s.find('x');
    if (x == std::string::npos) return false;
    try {
        w = std::stoi(s.substr(0, x));
        h = std::stoi(s.substr(x + 1));
    } catch (...) {
        return false;
    }
    return w > 0 && h > 0;
}

bool parseArgs(int argc, char** argv, Config& out, int* exitCode) {
    *exitCode = 0;
    out.outputDir = defaultOutputDir();

    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            std::cerr << "missing value for " << argv[i] << "\n";
            *exitCode = 2;
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return false;
        } else if (a == "--camera") {
            const char* v = need(i); if (!v) return false;
            if (!std::strcmp(v, "auto")) out.camera = CameraKind::Auto;
            else if (!std::strcmp(v, "picam")) out.camera = CameraKind::PiCam;
            else if (!std::strcmp(v, "webcam")) out.camera = CameraKind::Webcam;
            else { std::cerr << "unknown camera: " << v << "\n"; *exitCode = 2; return false; }
        } else if (a == "--output-dir") {
            const char* v = need(i); if (!v) return false;
            out.outputDir = v;
        } else if (a == "--webcam-index") {
            const char* v = need(i); if (!v) return false;
            out.webcamIndex = std::atoi(v);
        } else if (a == "--picam-name") {
            const char* v = need(i); if (!v) return false;
            out.picamName = v;
        } else if (a == "--rotate") {
            const char* v = need(i); if (!v) return false;
            out.rotate = std::atoi(v);
            if (out.rotate != 0 && out.rotate != 90 &&
                out.rotate != 180 && out.rotate != 270) {
                std::cerr << "bad --rotate (0|90|180|270): " << v << "\n";
                *exitCode = 2; return false;
            }
        } else if (a == "--camera-rotate") {
            const char* v = need(i); if (!v) return false;
            out.cameraRotate = std::atoi(v);
            if (out.cameraRotate != 0 && out.cameraRotate != 90 &&
                out.cameraRotate != 180 && out.cameraRotate != 270) {
                std::cerr << "bad --camera-rotate (0|90|180|270): " << v << "\n";
                *exitCode = 2; return false;
            }
        } else if (a == "--touch-rotate") {
            const char* v = need(i); if (!v) return false;
            out.touchRotate = std::atoi(v);
            if (out.touchRotate != 0 && out.touchRotate != 90 &&
                out.touchRotate != 180 && out.touchRotate != 270) {
                std::cerr << "bad --touch-rotate (0|90|180|270): " << v << "\n";
                *exitCode = 2; return false;
            }
        } else if (a == "--touch-flip-x") {
            out.touchFlipX = true;
        } else if (a == "--touch-flip-y") {
            out.touchFlipY = true;
        } else if (a == "--size") {
            const char* v = need(i); if (!v) return false;
            if (!parseSize(v, out.width, out.height)) {
                std::cerr << "bad --size (expected WxH): " << v << "\n"; *exitCode = 2; return false;
            }
        } else if (a == "--driver") {
            const char* v = need(i); if (!v) return false;
            out.driver = v;
        } else if (a == "--windowed") {
            out.windowed = true;
        } else if (a == "--face-model") {
            const char* v = need(i); if (!v) return false;
            out.faceModel = v;
        } else if (a == "--no-battery") {
            out.battery = false;
        } else if (a == "--battery-bus") {
            const char* v = need(i); if (!v) return false;
            out.batteryBus = std::atoi(v);
            if (out.batteryBus < 0) {
                std::cerr << "bad --battery-bus (expected /dev/i2c-N): " << v << "\n";
                *exitCode = 2; return false;
            }
        } else if (a == "--battery-shutdown") {
            out.batteryShutdown = true;
        } else {
            std::cerr << "unknown option: " << a << " (try --help)\n";
            *exitCode = 2;
            return false;
        }
    }

    // Create the output directory and any missing parents (e.g. ~/Pictures),
    // otherwise cv::imwrite has nowhere to write and captures are lost.
    ensureDir(out.outputDir);
    return true;
}

} // namespace olc
