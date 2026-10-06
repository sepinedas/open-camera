#pragma once

#include <string>
#include <vector>

namespace olc {

// Lists and navigates captured media (.jpg photos and .mp4 videos) in the
// output directory. Files are timestamp-named, so a plain name sort is
// chronological; newest is shown first.
class Gallery {
public:
    explicit Gallery(std::string dir) : dir_(std::move(dir)) { refresh(); }

    // Re-scan the directory, keeping the selection near where it was.
    void refresh();

    bool empty() const { return files_.empty(); }
    int count() const { return static_cast<int>(files_.size()); }
    int index() const { return index_; }

    const std::string& current() const { return files_[index_]; }
    bool currentIsVideo() const;

    // Step towards older or newer items. The strip reads like a timeline:
    // the past is to the left and the newest item is the right end, where the
    // gallery opens. The list does not wrap, so the left arrow at the oldest
    // photo (or the right one at the newest) stays put instead of jumping to
    // the far end.
    void older();
    void newer();
    bool atNewest() const { return index_ <= 0; }
    bool atOldest() const { return index_ >= count() - 1; }

    // Jump to the most recent item (files are stored newest-first).
    void selectNewest() { index_ = 0; }

    // Delete the current file from disk and re-scan. No-op when empty.
    void deleteCurrent();

    static bool isVideo(const std::string& path);

private:
    std::string dir_;
    std::vector<std::string> files_; // full paths, newest first
    int index_ = 0;
};

} // namespace olc
