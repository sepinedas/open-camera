#pragma once

#include <string>

namespace olc {

enum class CameraKind { Auto, PiCam, Webcam };

// Which cameras show a mirror image, like a phone's selfie camera. The Pi
// camera (the IMX500 AI camera on this build) faces the user, so by default it
// is the one mirrored.
enum class Mirror { None, PiCam, Webcam, All };

// Which Waveshare UPS HAT to expect. The models sit at different I2C addresses
// and carry different packs, so Auto simply probes for each in turn.
enum class UpsHat { Auto, B, D };

// Runtime options, populated from the command line (see config.cpp).
struct Config {
    CameraKind camera = CameraKind::Auto;
    std::string outputDir;   // where photos/videos are written (default: ~/Pictures/open-lego-camera)
    std::string driver;      // forced SDL_VIDEODRIVER ("kmsdrm", "fbcon", "x11", ...); empty = auto
    bool windowed = false;   // windowed instead of fullscreen (handy on a desktop)
    int webcamIndex = -1;    // force a specific /dev/videoN (-1 = probe)
    std::string picamName;   // libcamera camera-name to select when several exist
    int rotate = 0;          // rotate the whole UI 0/90/180/270 (clockwise)
    int cameraRotate = 0;    // rotate only the camera image 0/90/180/270 (clockwise)
    Mirror mirror = Mirror::PiCam; // flip these cameras left-right (preview + captures)
    int touchRotate = 0;     // rotate touch coords 0/90/180/270 to match the panel
    bool touchFlipX = false; // mirror touch horizontally
    bool touchFlipY = false; // mirror touch vertically
    int width = 1280;        // requested preview width
    int height = 720;        // requested preview height
    std::string faceModel;   // override path to the MediaPipe face_landmarker.task
    bool battery = true;     // look for a Waveshare UPS HAT battery gauge
    UpsHat batteryHat = UpsHat::Auto; // which model, or probe for either
    int batteryBus = 1;      // /dev/i2c-N the HAT sits on
    // Pack voltage at 0% / 100%, overriding the HAT's default curve. Both 0
    // means "use the board default" (see --battery-range).
    double batteryEmptyV = 0.0;
    double batteryFullV = 0.0;
    bool batteryShutdown = false; // power off when the pack reaches the cut-off
};

// Parse argv. Returns false and prints usage on --help or a bad flag; sets
// *exitCode accordingly so main() can `return`.
bool parseArgs(int argc, char** argv, Config& out, int* exitCode);

// Default capture directory: $HOME/Pictures/open-lego-camera (created if needed).
std::string defaultOutputDir();

// Create `path` and every missing parent (like `mkdir -p`). Returns true when
// the directory exists afterwards.
bool ensureDir(const std::string& path);

} // namespace olc
