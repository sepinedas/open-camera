// Deterministic preview of the 3D pig-face filter.
//
// Renders a simple synthetic head at several roll / turn angles and overlays
// the pig graphics via pig3d::render, writing a contact sheet to
// pig_preview.png. This lets the effect be eyeballed without a camera, and gives
// the geometry (landmark frame + 3D shading) a quick visual regression check.
//
// It draws straight through pig3d rather than through FaceFilter so the tool
// needs only OpenCV -- no MediaPipe install, no face model.
//
//   g++ -std=c++17 tools/pig_preview.cpp src/pig3d.cpp
//       $(pkg-config --cflags --libs opencv4) -o /tmp/pig_preview && /tmp/pig_preview
//
// Writes to argv[1] if given, else pig_preview.png in the working directory.

#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../src/pig3d.hpp"

using namespace olc;

namespace {

// Draw a plain skin-tone head with two eyes into `img`, rotated in-plane by
// `roll` radians and turned left/right by `yaw` (fakes a 3D turn by squashing
// the face horizontally and sliding the eyes off-centre). Returns the face box
// and the two eye centres so the pig can be anchored to them.
struct Head {
    cv::Rect box;
    cv::Point2f leftEye, rightEye;
};

Head drawHead(cv::Mat& img, cv::Point2f c, float faceH, float roll, float yaw) {
    const cv::Scalar skin(150, 190, 232); // BGR warm skin tone
    float faceW = faceH * 0.78f * (1.f - 0.35f * std::fabs(yaw));
    float ca = std::cos(roll), sa = std::sin(roll);
    auto rot = [&](cv::Point2f p) {
        return cv::Point2f(c.x + p.x * ca - p.y * sa, c.y + p.x * sa + p.y * ca);
    };
    // Head oval.
    cv::ellipse(img, c, cv::Size((int)(faceW * 0.5f), (int)(faceH * 0.5f)),
                roll * 180.0 / CV_PI, 0, 360, skin, cv::FILLED, cv::LINE_AA);
    // Eyes at ~0.42 face-height above centre, spread by the inter-ocular dist,
    // slid horizontally with the turn.
    float ipd = faceW * 0.46f;
    float ey = -0.16f * faceH;
    float shift = yaw * 0.18f * faceW;
    cv::Point2f le = rot({-ipd * 0.5f + shift, ey});
    cv::Point2f re = rot({ipd * 0.5f + shift, ey});
    for (cv::Point2f e : {le, re}) {
        cv::circle(img, e, std::max(3, (int)(faceW * 0.06f)), cv::Scalar(255, 255, 255),
                   cv::FILLED, cv::LINE_AA);
        cv::circle(img, e, std::max(2, (int)(faceW * 0.03f)), cv::Scalar(60, 45, 40),
                   cv::FILLED, cv::LINE_AA);
    }
    Head h;
    h.box = cv::Rect((int)(c.x - faceW * 0.5f), (int)(c.y - faceH * 0.5f),
                     (int)faceW, (int)faceH);
    h.leftEye = le;
    h.rightEye = re;
    return h;
}

} // namespace

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "pig_preview.png";
    const int cellW = 300, cellH = 340, cols = 4, rows = 2;
    cv::Mat sheet(cellH * rows, cellW * cols, CV_8UC3, cv::Scalar(60, 60, 60));

    struct Case { const char* label; float roll, yaw; bool eyes; };
    const Case cases[] = {
        {"front", 0.f, 0.f, true},        {"roll +20", 0.35f, 0.f, true},
        {"roll -20", -0.35f, 0.f, true},  {"roll +40", 0.70f, 0.f, true},
        {"turn left", 0.f, -0.6f, true},  {"turn right", 0.f, 0.6f, true},
        {"roll+turn", 0.3f, 0.45f, true}, {"no eyes (box)", 0.f, 0.f, false},
    };

    for (int i = 0; i < cols * rows; ++i) {
        int cx = (i % cols) * cellW, cy = (i / cols) * cellH;
        cv::Mat cell = sheet(cv::Rect(cx, cy, cellW, cellH));
        cell.setTo(cv::Scalar(205, 200, 195));
        const Case& t = cases[i];
        Head h = drawHead(cell, {cellW * 0.5f, cellH * 0.52f}, cellH * 0.5f, t.roll,
                          t.yaw);
        cv::Point2f le = t.eyes ? h.leftEye : cv::Point2f(-1, -1);
        cv::Point2f re = t.eyes ? h.rightEye : cv::Point2f(-1, -1);
        pig3d::render(cell, h.box, t.eyes, le, re, /*phase=*/0.0);
        cv::putText(cell, t.label, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(30, 30, 30), 2, cv::LINE_AA);
    }

    cv::imwrite(out, sheet);
    return 0;
}
