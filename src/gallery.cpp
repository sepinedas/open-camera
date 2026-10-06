#include "gallery.hpp"

#include <algorithm>
#include <cstdio>
#include <dirent.h>

namespace olc {

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

static bool hasExt(const std::string& name, const char* ext) {
    if (name.size() < std::string(ext).size()) return false;
    return lower(name.substr(name.size() - std::string(ext).size())) == ext;
}

bool Gallery::isVideo(const std::string& path) {
    return hasExt(path, ".mp4") || hasExt(path, ".avi") || hasExt(path, ".mov");
}

static bool isMedia(const std::string& name) {
    return hasExt(name, ".jpg") || hasExt(name, ".jpeg") ||
           hasExt(name, ".png") || Gallery::isVideo(name);
}

void Gallery::refresh() {
    std::string keep = files_.empty() ? std::string() : files_[index_];

    files_.clear();
    if (DIR* d = ::opendir(dir_.c_str())) {
        while (dirent* e = ::readdir(d)) {
            std::string name = e->d_name;
            if (name == "." || name == "..") continue;
            if (name.find(".video.mp4") != std::string::npos) continue; // in-progress mux temp
            if (name.find(".audio.wav") != std::string::npos) continue;
            if (isMedia(name)) files_.push_back(dir_ + "/" + name);
        }
        ::closedir(d);
    }
    // Timestamped names sort chronologically; reverse so newest is first.
    std::sort(files_.begin(), files_.end());
    std::reverse(files_.begin(), files_.end());

    // Restore selection to the same file if it still exists, else clamp.
    index_ = 0;
    if (!keep.empty()) {
        auto it = std::find(files_.begin(), files_.end(), keep);
        if (it != files_.end()) index_ = static_cast<int>(it - files_.begin());
    }
    if (index_ >= count()) index_ = std::max(0, count() - 1);
}

bool Gallery::currentIsVideo() const {
    return !empty() && isVideo(files_[index_]);
}

void Gallery::older() {
    if (!atOldest()) ++index_;
}

void Gallery::newer() {
    if (!atNewest()) --index_;
}

void Gallery::deleteCurrent() {
    if (empty()) return;
    ::remove(files_[index_].c_str());
    refresh();
}

} // namespace olc
