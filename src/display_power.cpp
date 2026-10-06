#include "display_power.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>

#include <SDL2/SDL.h>

#if defined(OLC_HAVE_LIBDRM)
#include <SDL2/SDL_syswm.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#endif

namespace olc {

namespace {

constexpr const char* kBacklightClass = "/sys/class/backlight";

std::string readTrimmed(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    if (!f || !std::getline(f, s)) return std::string();
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    return s;
}

// Only what a backlight's sysfs name and value can be, so a name handed to
// the shell below can never be anything but a path.
bool shellSafe(const std::string& s) {
    for (char c : s)
        if (!(std::isalnum((unsigned char)c) || c == '/' || c == '.' ||
              c == '_' || c == '-' || c == ':'))
            return false;
    return !s.empty();
}

// Write a sysfs attribute. The backlight class is root-only unless a udev
// rule opens it up (see the README), so a refused write is retried through
// `sudo -n`, which fails at once rather than prompting when sudo would need
// a password -- the same way the low-battery poweroff goes.
bool writeAttr(const std::string& path, const std::string& value) {
    {
        std::ofstream f(path);
        if (f && (f << value << '\n') && f.flush()) return true;
    }
    if (!shellSafe(path) || !shellSafe(value)) return false;
    const std::string cmd = "sudo -n sh -c 'echo " + value + " > " + path +
                            "' >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

std::vector<std::string> backlightDirs() {
    std::vector<std::string> out;
    if (DIR* d = ::opendir(kBacklightClass)) {
        while (dirent* e = ::readdir(d)) {
            if (e->d_name[0] == '.') continue;
            out.push_back(std::string(kBacklightClass) + "/" + e->d_name);
        }
        ::closedir(d);
    }
    return out;
}

#if defined(OLC_HAVE_LIBDRM)
// The DRM device SDL's kmsdrm driver opened and is master of. Using its fd
// rather than opening the card again is what makes this work without root:
// only the master may change a connector's state.
int drmFdOf(SDL_Window* win) {
#if SDL_VERSION_ATLEAST(2, 0, 15) && defined(SDL_VIDEO_DRIVER_KMSDRM)
    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    if (win && SDL_GetWindowWMInfo(win, &info) &&
        info.subsystem == SDL_SYSWM_KMSDRM)
        return info.info.kmsdrm.drm_fd;
#else
    (void)win;
#endif
    return -1;
}

// Set DPMS on every connected connector. Returns how many took it.
int setDpms(int fd, uint64_t mode) {
    int n = 0;
    drmModeRes* res = drmModeGetResources(fd);
    if (!res) return 0;
    for (int i = 0; i < res->count_connectors; ++i) {
        drmModeConnector* c = drmModeGetConnector(fd, res->connectors[i]);
        if (!c) continue;
        if (c->connection == DRM_MODE_CONNECTED) {
            for (int k = 0; k < c->count_props; ++k) {
                drmModePropertyRes* p = drmModeGetProperty(fd, c->props[k]);
                if (!p) continue;
                if (std::strcmp(p->name, "DPMS") == 0 &&
                    drmModeConnectorSetProperty(fd, c->connector_id, p->prop_id,
                                                mode) == 0)
                    ++n;
                drmModeFreeProperty(p);
            }
        }
        drmModeFreeConnector(c);
    }
    drmModeFreeResources(res);
    return n;
}
#endif

// The legacy firmware display stack's own switch. On a KMS system the command
// still exists and still exits 0 while doing nothing, so success is judged by
// what it reports back, not by its exit status.
bool vcgencmdDisplayPower(bool on) {
    const std::string cmd = std::string("vcgencmd display_power ") +
                            (on ? "1" : "0") + " 2>/dev/null";
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) return false;
    char buf[128] = {};
    const bool got = std::fgets(buf, sizeof(buf), p) != nullptr;
    ::pclose(p);
    return got && std::strstr(buf, on ? "display_power=1" : "display_power=0");
}

} // namespace

std::string DisplayPower::sleep(SDL_Window* win) {
    std::string did;
    auto note = [&](const std::string& s) { did += (did.empty() ? "" : ", ") + s; };

    // Backlight first: it is the bulk of an LCD's power, and on most panels
    // the only part of it that can actually be switched off. bl_power 4 is
    // FB_BLANK_POWERDOWN; brightness 0 as well, for the drivers that only
    // honour that.
    backlights_.clear();
    for (const std::string& dir : backlightDirs()) {
        Backlight b{dir, readTrimmed(dir + "/brightness"),
                    readTrimmed(dir + "/bl_power")};
        const bool off = writeAttr(dir + "/bl_power", "4") |
                         writeAttr(dir + "/brightness", "0");
        if (off) {
            backlights_.push_back(b);
            note("backlight " + dir.substr(std::strlen(kBacklightClass) + 1));
        }
    }

    dpms_ = false;
#if defined(OLC_HAVE_LIBDRM)
    const int fd = drmFdOf(win);
    if (fd >= 0 && setDpms(fd, DRM_MODE_DPMS_OFF) > 0) {
        dpms_ = true;
        note("display pipeline (DPMS)");
    }
#else
    (void)win;
#endif

    // Only where neither of the above found anything to switch: on a KMS
    // system it would do nothing, and on the legacy stack they do not exist.
    vcgencmd_ = false;
    if (backlights_.empty() && !dpms_ && vcgencmdDisplayPower(false)) {
        vcgencmd_ = true;
        note("display (vcgencmd)");
    }
    return did;
}

void DisplayPower::wake(SDL_Window* win) {
    // The pipeline before the light, so the backlight comes up on a picture
    // rather than on whatever the panel shows while it starts scanning.
#if defined(OLC_HAVE_LIBDRM)
    if (dpms_) {
        const int fd = drmFdOf(win);
        if (fd >= 0) setDpms(fd, DRM_MODE_DPMS_ON);
    }
#else
    (void)win;
#endif
    dpms_ = false;
    if (vcgencmd_) vcgencmdDisplayPower(true);
    vcgencmd_ = false;

    for (const Backlight& b : backlights_) {
        writeAttr(b.dir + "/bl_power", b.blPower.empty() ? "0" : b.blPower);
        // Asleep at zero brightness would wake to a dark screen, so a saved
        // zero (or none) comes back at full instead.
        std::string level = b.brightness;
        if (level.empty() || level == "0") level = readTrimmed(b.dir + "/max_brightness");
        if (!level.empty()) writeAttr(b.dir + "/brightness", level);
    }
    backlights_.clear();
}

} // namespace olc
