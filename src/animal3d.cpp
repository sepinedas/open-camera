#include "animal3d.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace olc::animal3d {

namespace {

using cv::Vec3f;
using cv::Matx33f;

constexpr float kPi = 3.14159265358979f;
// Supersampling factor for anti-aliasing. This stays at 2 even on a Pi 5, and
// the limit is memory bandwidth rather than arithmetic: the colour layer,
// coverage mask and z-buffer come to 20 bytes per supersample, so a pig
// covering a 600x700 region already allocates and clears ~34 MB per frame
// (~1 GB/s at 30 fps). Raising this to 3 would put that past 75 MB per frame,
// a meaningful slice of the board's ~17 GB/s, to buy a barely visible
// improvement on the ear silhouettes.
constexpr int kSS = 2;
constexpr float kCamZ = 7.0f;  // camera distance in model units (eye-widths)

float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

Vec3f norm(const Vec3f& v) {
    float n = std::sqrt(v.dot(v));
    return n > 1e-8f ? v * (1.f / n) : v;
}

// --- Mesh ------------------------------------------------------------------

// A triangle mesh with per-vertex position/normal/colour and a single material.
// `doubleSided` ears are lit from whichever face points at the camera; the
// closed snout is back-face culled instead.
struct Mesh {
    std::vector<Vec3f> pos;    // model space
    std::vector<Vec3f> nrm;    // model-space vertex normals (accumulated)
    std::vector<Vec3f> col;    // BGR 0..255 base colour
    std::vector<cv::Vec3i> tri;
    bool doubleSided = false;
    float ambient = 0.35f;
    float spec = 0.2f;
    float shin = 14.f;

    int add(const Vec3f& p, const Vec3f& c) {
        pos.push_back(p);
        col.push_back(c);
        nrm.emplace_back(0.f, 0.f, 0.f);
        return (int)pos.size() - 1;
    }
    void face(int a, int b, int c) { tri.emplace_back(a, b, c); }

    // Smooth vertex normals by accumulating adjacent face normals.
    void computeNormals() {
        for (auto& n : nrm) n = Vec3f(0.f, 0.f, 0.f);
        for (const auto& t : tri) {
            Vec3f fn = (pos[t[1]] - pos[t[0]]).cross(pos[t[2]] - pos[t[0]]);
            nrm[t[0]] += fn;
            nrm[t[1]] += fn;
            nrm[t[2]] += fn;
        }
        for (auto& n : nrm) n = norm(n);
    }
};

// Rotate a point about an axis-aligned pivot by a rotation matrix.
Vec3f rotAbout(const Matx33f& R, const Vec3f& p, const Vec3f& pivot) {
    return R * (p - pivot) + pivot;
}

Matx33f rotZ(float a) {
    float c = std::cos(a), s = std::sin(a);
    return Matx33f(c, -s, 0, s, c, 0, 0, 0, 1);
}

// --- Mesh builders ---------------------------------------------------------

// An orthonormal (u, v) basis spanning the plane perpendicular to `axis`.
void basis(const Vec3f& axis, Vec3f& u, Vec3f& v) {
    Vec3f up(0.f, 1.f, 0.f);
    if (std::fabs(axis.dot(up)) > 0.9f) up = Vec3f(1.f, 0.f, 0.f);
    u = norm(up.cross(axis)); // roughly head-right
    v = norm(axis.cross(u));  // roughly head-up/down
}

// Everything that differs between the animals. The rig around it -- pose,
// camera, z-buffer, shading -- is shared, so adding a species is a row here
// rather than another renderer.
//
// Lengths and radii are in eye-separation units; ear corners are multiples of
// the measured half-head-width and offsets from the measured crown. Colours
// are BGR, to match the frame they are composited into.
struct Style {
    // --- muzzle ---
    float muzzleLen;   // how far it protrudes along the face normal
    // How far the muzzle axis tilts *down* the face as it comes forward. This
    // matters more than it looks: a muzzle that only protrudes is seen end-on
    // from the front, where a tapering one hides behind its own nose and shows
    // a few pixels of rim. Dropping the axis presents it side-on instead.
    float muzzleDrop;
    float muzzleR0;    // radius at the face, as a multiple of the alar width
    float muzzleTaper; // tip radius / base radius: >1 flares, <1 narrows
    float muzzleFlat;  // cross-section height / width
    // padCol is the flat front face of the muzzle; noseCol is the nose itself
    // (the pig's nostrils, the dog's leather). Painting the front face with the
    // nose colour turns the whole muzzle into one dark blob.
    Vec3f muzzleCol, padCol, noseCol;
    bool nostrils;     // pig: two holes in a flat disc; dog: a domed nose

    // --- ears ---
    // An ear is a lobe swept from an attachment on the head to a tip, with a
    // width profile along the way. x is a multiple of headHalfW, y an offset
    // from crownY.
    float earAttachX, earAttachY; // centre of the attachment on the head
    float earTipX, earTipY;       // centre of the far end
    float earHalfW;               // widest half-width, multiple of headHalfW
    float earRound;               // tip shape: small = round lobe, large = point
    float earBulge, earCurl;
    Vec3f earCol, earInnerCol;
};

Style styleFor(Species sp) {
    Style s{};
    if (sp == Species::Pig) {
        s.muzzleLen = 0.58f;
        s.muzzleDrop = 0.12f;
        s.muzzleR0 = 1.85f;
        s.muzzleTaper = 1.12f; // a pig's snout flares toward the disc
        s.muzzleFlat = 0.80f;  // wider than tall
        s.muzzleCol = Vec3f(168, 152, 236);
        s.padCol = Vec3f(184, 168, 243);
        s.noseCol = Vec3f(40, 32, 70);
        s.nostrils = true;
        // Stands ON the crown, leaning up and outward. The tip's y is
        // *negative* -- above the crown -- which is what keeps the whole ear
        // clear of the face.
        s.earAttachX = 0.56f; s.earAttachY = 0.12f;
        s.earTipX = 0.90f;    s.earTipY = -0.86f;
        s.earHalfW = 0.39f;   s.earRound = 2.4f;
        s.earBulge = 0.16f;   s.earCurl = 0.10f;
        s.earCol = Vec3f(172, 152, 239);
        s.earInnerCol = Vec3f(122, 98, 202);
    } else { // Dog
        // A longer muzzle that narrows to a small domed nose, and big soft
        // ears that hang down past the eye line instead of standing up.
        // A broad rounded muzzle pad rather than a long tube. A long muzzle is
        // more dog-like in profile but the camera looks at a face head-on,
        // where it is seen end-on and collapses behind its own nose; dropping
        // the axis far enough to fix that put it below the chin, reading as a
        // beard. A short wide bulge carrying a big dark nose is what actually
        // reads as a dog from the front.
        s.muzzleLen = 0.45f;
        s.muzzleDrop = 0.17f;
        s.muzzleR0 = 2.40f;
        s.muzzleTaper = 0.85f; // barely tapers: stays broad at the nose
        s.muzzleFlat = 0.82f;
        // Brown, near the ear colour. A pale muzzle is the more typical dog
        // marking but it does not read here: cream sits within ~25 levels of
        // human skin tone once shaded, so the whole muzzle vanished into the
        // face. Contrast has to go *darker* than skin, not lighter.
        s.muzzleCol = Vec3f(105, 148, 196); // mid brown: clear of skin and of
                                            // the near-black nose above it
        s.padCol = Vec3f(124, 168, 214);    // muzzle front, a shade lighter
        s.noseCol = Vec3f(48, 42, 40);     // near-black nose leather
        s.nostrils = false;
        // A broad soft lobe hanging down the *side* of the head. It attaches
        // near the top corner and drapes outside the silhouette, so it never
        // crosses the eye, and its tip is rounded rather than pointed.
        s.earAttachX = 0.94f; s.earAttachY = -0.06f;
        s.earTipX = 1.24f;    s.earTipY = 1.28f;
        s.earHalfW = 0.37f;   s.earRound = 9.0f;
        s.earBulge = 0.15f;   s.earCurl = 0.12f;
        s.earCol = Vec3f(62, 96, 138);
        s.earInnerCol = Vec3f(78, 92, 126);
    }
    return s;
}

// Where the snout sits and how big it is, derived from the measured nose so
// the tube lands *on* the nose instead of at a fixed spot on the face. The
// front pad is centred on the nose tip: `base` is pulled back along the axis by
// the snout's length so protruding forward puts the pad there. Shared with
// buildNostrils so the holes cannot drift off the pad.
struct SnoutFrame {
    Vec3f base, axis, u, v, padCentre;
    float len, rx, ry;
};

SnoutFrame snoutFrame(const Proportions& p, const Style& st) {
    SnoutFrame s;
    s.axis = norm(Vec3f(0.f, st.muzzleDrop, -1.f)); // forward and downward
    s.len = st.muzzleLen;
    // An animal muzzle is chunkier than the human nose underneath it; scale
    // the measured alar half-width up rather than inventing an absolute size.
    s.rx = st.muzzleR0 * p.noseHalfW;
    s.ry = st.muzzleFlat * s.rx;
    // The base sits on the face at the measured nose height and the tube
    // protrudes forward from there, so the pad ends up just in front of and
    // just below the nose -- where a snout actually is. (Pinning the *pad* to
    // the nose instead would push the base back behind the eye plane and bury
    // most of the snout inside the head.)
    s.base = Vec3f(0.f, p.noseY, -0.28f);
    s.padCentre = s.base + s.axis * s.len;
    basis(s.axis, s.u, s.v);
    return s;
}

// The snout: an elliptical tube protruding forward (-Z) from the face, capped by
// a domed front pad. Wider than tall, flaring slightly toward the front.
Mesh buildMuzzle(const Proportions& prop, const Style& st) {
    Mesh m;
    m.ambient = 0.42f;
    m.spec = 0.26f;
    m.shin = 18.f;
    const Vec3f pink = st.muzzleCol;
    const SnoutFrame sf = snoutFrame(prop, st);
    const Vec3f base = sf.base;
    const Vec3f axis = sf.axis;
    const float len = sf.len;
    const int nSeg = 26, nRing = 4;
    const Vec3f u = sf.u, v = sf.v;

    // Rings from the face out to the front.
    std::vector<std::vector<int>> ring(nRing);
    for (int r = 0; r < nRing; ++r) {
        float t = (float)r / (nRing - 1);
        Vec3f c = base + axis * (t * len);
        // Interpolate base radius -> tip radius, so a pig flares out and a
        // dog narrows toward the nose.
        const float k = 1.f + (st.muzzleTaper - 1.f) * t;
        float rx = sf.rx * k;
        float ry = sf.ry * k;
        for (int i = 0; i < nSeg; ++i) {
            float a = 2.f * kPi * i / nSeg;
            Vec3f p = c + u * (rx * std::cos(a)) + v * (ry * std::sin(a));
            ring[r].push_back(m.add(p, pink));
        }
    }
    for (int r = 0; r + 1 < nRing; ++r)
        for (int i = 0; i < nSeg; ++i) {
            int i2 = (i + 1) % nSeg;
            m.face(ring[r][i], ring[r][i2], ring[r + 1][i2]);
            m.face(ring[r][i], ring[r + 1][i2], ring[r + 1][i]);
        }

    // Front pad: a slightly brighter, domed disc closing the tube. `axis` points
    // toward the camera (its z is negative), so adding it bulges the pad OUT.
    const Vec3f pad = st.padCol;
    Vec3f fc = base + axis * len;
    float frx = sf.rx * st.muzzleTaper, fry = sf.ry * st.muzzleTaper;
    int centre = m.add(fc + axis * 0.07f, pad); // dome bulging toward camera
    std::vector<int> fringe;
    for (int i = 0; i < nSeg; ++i) {
        float a = 2.f * kPi * i / nSeg;
        Vec3f p = fc + u * (frx * std::cos(a)) + v * (fry * std::sin(a));
        fringe.push_back(m.add(p, pad));
    }
    for (int i = 0; i < nSeg; ++i)
        m.face(centre, fringe[(i + 1) % nSeg], fringe[i]);

    m.computeNormals();
    return m;
}

// Two nostrils: small dark domes recessed into the snout's front pad.
Mesh buildNose(const Proportions& prop, const Style& st) {
    Mesh m;
    m.doubleSided = true; // tiny discs; skip culling so winding never hides them
    m.ambient = 0.34f;
    m.spec = 0.05f;
    m.shin = 20.f;
    const Vec3f dark = st.noseCol;
    // Same frame as the muzzle -- these used to repeat its constants, so any
    // change to the muzzle silently left the nostrils behind.
    const SnoutFrame sf = snoutFrame(prop, st);
    const Vec3f axis = sf.axis;
    const Vec3f u = sf.u, v = sf.v; // u ~ head-right (image), v ~ head-down
    const Vec3f fc = sf.padCentre;

    const int nSeg = 16;

    if (!st.nostrils) {
        // Dog: one rounded nose bulging off the end of the muzzle, rather than
        // a pig's flat disc with two holes punched in it. Built as a hemisphere
        // of latitude rings so it catches a highlight and reads as wet leather.
        m.doubleSided = false; // closed and convex: back faces can be culled
        m.spec = 0.45f;
        m.shin = 30.f;
        const float R = 0.74f * sf.rx * st.muzzleTaper;
        const Vec3f centre = fc + axis * (0.35f * R);
        const int nRing = 6;
        std::vector<std::vector<int>> ring(nRing);
        for (int r = 0; r < nRing; ++r) {
            // 0 at the equator (against the muzzle), 1 at the pole (forward).
            const float lat = 0.5f * kPi * (float)r / (nRing - 1);
            const float cr = std::cos(lat) * R, cz = std::sin(lat) * R;
            for (int i = 0; i < nSeg; ++i) {
                const float a = 2.f * kPi * i / nSeg;
                // Slightly wider than tall, like a real nose leather.
                Vec3f pt = centre + u * (1.15f * cr * std::cos(a)) +
                           v * (0.90f * cr * std::sin(a)) + axis * cz;
                ring[r].push_back(m.add(pt, st.noseCol));
            }
        }
        // Wound so the outward face is the front face. The tube above sweeps
        // its rings the other way round its axis, so copying its winding here
        // put the whole dome back-facing and culling swallowed it, leaving
        // only a sliver of rim showing past the muzzle.
        for (int r = 0; r + 1 < nRing; ++r)
            for (int i = 0; i < nSeg; ++i) {
                const int j = (i + 1) % nSeg;
                m.face(ring[r][i], ring[r + 1][j], ring[r][j]);
                m.face(ring[r][i], ring[r + 1][i], ring[r + 1][j]);
            }
        m.computeNormals();
        return m;
    }

    for (float side : {-1.f, 1.f}) {
        // Two prominent holes near the centre of the domed pad, sitting just in
        // front of it so they win the z-test; slanted outward as a pig's are.
        // (+axis moves toward the camera; +v is downward.)
        Vec3f c = fc + u * (0.38f * sf.rx * st.muzzleTaper * side) +
                  v * 0.015f + axis * 0.09f;
        float rx = 0.21f * sf.rx * st.muzzleTaper;
        float ry = 0.30f * sf.ry * st.muzzleTaper;
        float ca = std::cos(side * 0.28f), sa = std::sin(side * 0.28f);
        int centre = m.add(c + axis * 0.02f, dark);
        std::vector<int> fr;
        for (int i = 0; i < nSeg; ++i) {
            float a = 2.f * kPi * i / nSeg;
            float ex = rx * std::cos(a), ey = ry * std::sin(a);
            float rxr = ex * ca - ey * sa, ryr = ex * sa + ey * ca; // tilt
            fr.push_back(m.add(c + u * rxr + v * ryr, dark));
        }
        for (int i = 0; i < nSeg; ++i)
            m.face(centre, fr[i], fr[(i + 1) % nSeg]);
    }
    m.computeNormals();
    return m;
}

// One ear: a broad, floppy pig ear that attaches along the crown and drapes
// OUTWARD and DOWN to a soft tip out by the outer eye, tilted forward so its
// front shows -- exactly a domestic pig's folded ear, not an upright horn.
//
// Built from three corners so it stays a clean flap (no self-folding crease):
//   A = inner-top attachment (near the crown, just inside the eye line)
//   B = outer-top attachment (crown, further out)
//   C = the drooping tip, out to the side and down near eye level
// The surface is the triangle A-B-C: each row t interpolates from the top edge
// (A..B) toward the tip C, so it is wide along the crown and narrows to the tip.
// A gentle mid-surface bulge plus a slight forward curl of the tip give it body,
// and the lower-inner region is tinted a deeper pink for the ear's hollow.
Mesh buildEar(float side, float wiggle, const Proportions& prop, const Style& st) {
    Mesh m;
    m.doubleSided = true;
    m.ambient = 0.38f;
    m.spec = 0.12f;
    m.shin = 12.f;
    const Vec3f pink = st.earCol;
    const Vec3f inner = st.earInnerCol;

    // An ear is a lobe swept along an axis from where it joins the head to its
    // tip, widening out of the attachment and closing again at the far end.
    // Sweeping a *width profile* is what makes it read as an ear: blending the
    // whole outline to a single point, as this used to, yields a flat triangle
    // that looks like a horn however it is positioned.
    //
    // Model eyes are at (+-0.5, -0.35); +y is down, -z is toward the camera.
    const float hw = prop.headHalfW;
    const float cy = prop.crownY;
    const Vec3f attach(side * st.earAttachX * hw, cy + st.earAttachY, -0.08f);
    const Vec3f tip(side * st.earTipX * hw, cy + st.earTipY, -0.26f);
    const Vec3f axis = tip - attach;
    // Across the lobe, perpendicular to its axis and to the view direction, so
    // the width is always spread across the screen rather than into it.
    Vec3f across = norm(Vec3f(0.f, 0.f, -1.f).cross(norm(axis)));
    if (across[0] * side < 0.f) across = -across; // keep +across pointing outward
    Vec3f Nrm = norm(across.cross(norm(axis)));
    if (Nrm[2] > 0.f) Nrm = -Nrm; // face the camera
    const float maxHalfW = st.earHalfW * hw;
    const float bulge = st.earBulge, tipCurl = st.earCurl;

    // Half-width along the lobe: broad where it meets the head, widest a third
    // of the way along, then closing. `earRound` sets how abruptly it closes --
    // a small value rounds the tip off, a large one draws it to a point.
    auto halfWidth = [&](float t) {
        const float body = 0.70f + 0.30f * std::sin(kPi * clampf(t * 0.85f + 0.08f, 0.f, 1.f));
        const float close = std::pow(std::max(0.f, 1.f - std::pow(t, st.earRound)), 0.45f);
        return maxHalfW * body * close;
    };

    const int nA = 13, nT = 14;
    std::vector<std::vector<int>> g(nT, std::vector<int>(nA));
    for (int ti = 0; ti < nT; ++ti) {
        const float t = (float)ti / (nT - 1); // 0 at the head, 1 at the tip
        const Vec3f centre = attach + axis * t;
        const float w = halfWidth(t);
        for (int ai = 0; ai < nA; ++ai) {
            const float a = -1.f + 2.f * (float)ai / (nA - 1); // -1..1 across
            Vec3f p = centre + across * (w * a);
            // Soft body: bulge the middle toward the camera, easing out at the
            // rim so the lobe has thickness instead of reading as paper.
            const float edge = std::cos(a * 0.5f * kPi);
            p += Nrm * (bulge * edge * (1.f - 0.35f * t));
            // A gentle forward curl toward the tip, so it hangs rather than
            // standing perfectly flat.
            p += Nrm * (tipCurl * t * t);
            // Darker inner hollow down the centre of the lobe.
            const float inF = clampf((t - 0.18f) * 1.5f, 0.f, 1.f) *
                              clampf(1.f - std::fabs(a) * 1.5f, 0.f, 1.f);
            const Vec3f col = pink * (1.f - inF) + inner * inF;
            g[ti][ai] = m.add(p, col);
        }
    }
    for (int ti = 0; ti + 1 < nT; ++ti)
        for (int ai = 0; ai + 1 < nA; ++ai) {
            m.face(g[ti][ai], g[ti][ai + 1], g[ti + 1][ai + 1]);
            m.face(g[ti][ai], g[ti + 1][ai + 1], g[ti + 1][ai]);
        }

    // Idle wiggle: rock the whole ear about its attachment in the image plane.
    if (wiggle != 0.f) {
        Matx33f Rw = rotZ(side * wiggle);
        Vec3f pivot = attach;
        for (auto& p : m.pos) p = rotAbout(Rw, p, pivot);
    }
    m.computeNormals();
    return m;
}

// --- Pose + rasteriser -----------------------------------------------------

struct Pose {
    Matx33f R;         // model -> world rotation
    cv::Point2f anchor; // image point the model origin projects to
    float scale;        // model unit -> pixels at the head's depth
};

// Estimate head pose from the measured head landmarks and the face box. The
// eye line fixes roll and the in-plane scale. Yaw/pitch are taken from `head`
// when the caller measured them from the face mesh; otherwise they fall back to
// the old approximation from where the eyes sit inside the box. Model eyes sit
// at (+-0.5, -0.35, -0.2), so the model origin is the head centre and the whole
// rig rotates about it.
Pose estimatePose(const cv::Rect& f, const Head& head) {
    float roll = 0.f, yaw = 0.f, pitch = 0.f, eyeDist;
    cv::Point2f eyeMid;
    if (head.hasEyes) {
        cv::Point2f d = head.rightEye - head.leftEye;
        eyeDist = std::sqrt(d.x * d.x + d.y * d.y);
        eyeMid = (head.leftEye + head.rightEye) * 0.5f;
        roll = std::atan2(d.y, d.x);
        if (head.hasPose) {
            yaw = clampf(head.yaw, -0.95f, 0.95f);
            pitch = clampf(head.pitch, -0.6f, 0.6f);
        } else {
            float bcx = f.x + f.width * 0.5f;
            float nx = (eyeMid.x - bcx) / (0.5f * f.width);
            float ny = (eyeMid.y - (f.y + 0.45f * f.height)) / (0.5f * f.height);
            yaw = clampf(nx * 2.2f, -0.9f, 0.9f);    // + => turned toward image-right
            pitch = clampf(-ny * 1.1f, -0.5f, 0.5f); // + => chin up
        }
    } else {
        eyeDist = 0.42f * f.width;
        eyeMid = cv::Point2f(f.x + 0.5f * f.width, f.y + 0.42f * f.height);
        if (head.hasPose) {
            yaw = clampf(head.yaw, -0.95f, 0.95f);
            pitch = clampf(head.pitch, -0.6f, 0.6f);
        }
    }

    // R = Rz(roll) * Ry(yaw) * Rx(pitch), applied to model points.
    float cr = std::cos(roll), sr = std::sin(roll);
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    Matx33f Rz(cr, -sr, 0, sr, cr, 0, 0, 0, 1);
    Matx33f Ry(cy, 0, sy, 0, 1, 0, -sy, 0, cy);
    Matx33f Rx(1, 0, 0, 0, cp, -sp, 0, sp, cp);

    Pose ps;
    ps.R = Rz * Ry * Rx;

    // Calibrate scale so the model eye separation matches the observed one, and
    // anchor so the model eye-midpoint lands on the observed eye-midpoint. This
    // undoes the yaw/perspective foreshortening baked into the observation.
    Vec3f eyeMidModel(0.f, -0.35f, -0.2f);
    Vec3f wMid = ps.R * eyeMidModel;
    float perspMid = kCamZ / (kCamZ + wMid[2]);
    Vec3f eyeAxis = ps.R * Vec3f(1.f, 0.f, 0.f); // model eye separation vector
    float fe = clampf(std::sqrt(eyeAxis[0] * eyeAxis[0] + eyeAxis[1] * eyeAxis[1]),
                      0.3f, 1.f);
    ps.scale = eyeDist / (perspMid * fe);
    ps.anchor = eyeMid - cv::Point2f(ps.scale * perspMid * wMid[0],
                                     ps.scale * perspMid * wMid[1]);
    return ps;
}

// Project a model point to screen (in supersampled pixels) plus its camera
// depth. Perspective divide about the camera at (0,0,-kCamZ) looking +Z.
struct Proj { cv::Point2f s; float depth; };
Proj project(const Pose& ps, const Vec3f& world, float ssScale, cv::Point2f ssOrg) {
    float depth = kCamZ + world[2];
    float persp = kCamZ / std::max(0.5f, depth);
    cv::Point2f img = ps.anchor + cv::Point2f(ps.scale * persp * world[0],
                                              ps.scale * persp * world[1]);
    Proj p;
    p.s = (img - ssOrg) * ssScale;
    p.depth = depth;
    return p;
}

// Light rig (camera space): key light from the upper-left front.
const Vec3f kLight = norm(Vec3f(-0.4f, -0.55f, -0.8f));
const Vec3f kHalf = norm(kLight + Vec3f(0.f, 0.f, -1.f)); // view = -Z

// Rasterise one mesh into the supersampled layer with a shared z-buffer.
// `layer` is BGR float, `cover` marks written pixels, `zbuf` resolves occlusion.
void raster(const Mesh& m, const Pose& ps, float ssScale, cv::Point2f ssOrg,
            cv::Mat& layer, cv::Mat& cover, cv::Mat& zbuf) {
    const int W = layer.cols, H = layer.rows;
    std::vector<Vec3f> world(m.pos.size()), wnrm(m.nrm.size());
    std::vector<Proj> pr(m.pos.size());
    for (size_t i = 0; i < m.pos.size(); ++i) {
        world[i] = ps.R * m.pos[i];
        wnrm[i] = ps.R * m.nrm[i];
        pr[i] = project(ps, world[i], ssScale, ssOrg);
    }

    for (const auto& t : m.tri) {
        const Proj &A = pr[t[0]], &B = pr[t[1]], &C = pr[t[2]];
        // Signed area in screen space (y-down): >0 is a front face.
        float area = (B.s.x - A.s.x) * (C.s.y - A.s.y) -
                     (C.s.x - A.s.x) * (B.s.y - A.s.y);
        if (std::fabs(area) < 1e-4f) continue;
        bool back = area < 0.f;
        if (back && !m.doubleSided) continue; // cull for the closed snout

        int minx = (int)std::floor(std::min({A.s.x, B.s.x, C.s.x}));
        int maxx = (int)std::ceil(std::max({A.s.x, B.s.x, C.s.x}));
        int miny = (int)std::floor(std::min({A.s.y, B.s.y, C.s.y}));
        int maxy = (int)std::ceil(std::max({A.s.y, B.s.y, C.s.y}));
        minx = std::max(0, minx); miny = std::max(0, miny);
        maxx = std::min(W - 1, maxx); maxy = std::min(H - 1, maxy);
        float invArea = 1.f / area;

        for (int y = miny; y <= maxy; ++y) {
            float* zb = zbuf.ptr<float>(y);
            cv::Vec3f* lp = layer.ptr<cv::Vec3f>(y);
            float* cp = cover.ptr<float>(y);
            for (int x = minx; x <= maxx; ++x) {
                float px = x + 0.5f, py = y + 0.5f;
                // Barycentric weights.
                float w0 = ((B.s.x - px) * (C.s.y - py) -
                            (C.s.x - px) * (B.s.y - py)) * invArea;
                float w1 = ((C.s.x - px) * (A.s.y - py) -
                            (A.s.x - px) * (C.s.y - py)) * invArea;
                float w2 = 1.f - w0 - w1;
                if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;

                float depth = w0 * A.depth + w1 * B.depth + w2 * C.depth;
                if (depth >= zb[x]) continue; // farther than what's there

                Vec3f n = norm(w0 * wnrm[t[0]] + w1 * wnrm[t[1]] + w2 * wnrm[t[2]]);
                if (n[2] > 0.f) n = -n; // face the camera (both sides / robustness)
                Vec3f base = w0 * m.col[t[0]] + w1 * m.col[t[1]] + w2 * m.col[t[2]];

                float diff = std::max(0.f, n.dot(kLight));
                float shade = m.ambient + (1.f - m.ambient) * diff;
                float s = m.spec * std::pow(std::max(0.f, n.dot(kHalf)), m.shin);
                Vec3f c = base * shade + Vec3f(255, 255, 255) * s;

                zb[x] = depth;
                lp[x] = cv::Vec3f(std::min(255.f, c[0]), std::min(255.f, c[1]),
                                  std::min(255.f, c[2]));
                cp[x] = 1.f;
            }
        }
    }
}

} // namespace

void render(cv::Mat& frame, const cv::Rect& face, bool hasEyes,
            cv::Point2f leftEye, cv::Point2f rightEye, double phase,
            Species species) {
    Head h;
    h.hasEyes = hasEyes;
    h.leftEye = leftEye;
    h.rightEye = rightEye;
    render(frame, face, h, phase, species);
}

void render(cv::Mat& frame, const cv::Rect& face, const Head& head, double phase,
            Species species) {
    if (frame.empty() || frame.type() != CV_8UC3) return;
    if (face.width < 40 || face.height < 40) return;

    Pose ps = estimatePose(face, head);
    if (ps.scale < 8.f) return;

    float wig = 0.05f * std::sin((float)phase * 0.11f);
    std::vector<Mesh> meshes;
    const Style st = styleFor(species);
    meshes.push_back(buildEar(-1.f, wig, head.prop, st));
    meshes.push_back(buildEar(+1.f, wig, head.prop, st));
    meshes.push_back(buildMuzzle(head.prop, st));
    meshes.push_back(buildNose(head.prop, st));

    // Image-space bounding box of every projected vertex -> the region we touch.
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    for (const Mesh& m : meshes)
        for (const Vec3f& p : m.pos) {
            Proj pr = project(ps, ps.R * p, 1.f, cv::Point2f(0, 0));
            minx = std::min(minx, pr.s.x); maxx = std::max(maxx, pr.s.x);
            miny = std::min(miny, pr.s.y); maxy = std::max(maxy, pr.s.y);
        }
    cv::Rect roi((int)std::floor(minx) - 2, (int)std::floor(miny) - 2,
                 (int)std::ceil(maxx - minx) + 4, (int)std::ceil(maxy - miny) + 4);
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (roi.width < 2 || roi.height < 2) return;

    // Supersampled layer, coverage mask and z-buffer over the ROI only.
    cv::Point2f org(roi.x, roi.y);
    int LW = roi.width * kSS, LH = roi.height * kSS;
    cv::Mat layer(LH, LW, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat cover(LH, LW, CV_32F, cv::Scalar(0));
    cv::Mat zbuf(LH, LW, CV_32F, cv::Scalar(1e9f));
    for (const Mesh& m : meshes)
        raster(m, ps, (float)kSS, org, layer, cover, zbuf);

    // Downsample (box filter) to resolve the supersampled edges, then composite
    // the pig over the frame ROI with the resulting per-pixel coverage.
    cv::Mat colDown, covDown;
    cv::resize(layer, colDown, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_AREA);
    cv::resize(cover, covDown, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_AREA);

    cv::Mat dst = frame(roi);
    for (int y = 0; y < roi.height; ++y) {
        cv::Vec3b* d = dst.ptr<cv::Vec3b>(y);
        const cv::Vec3f* c = colDown.ptr<cv::Vec3f>(y);
        const float* a = covDown.ptr<float>(y);
        for (int x = 0; x < roi.width; ++x) {
            float cov = a[x];
            if (cov <= 0.003f) continue;
            // colDown is the coverage-weighted colour sum from INTER_AREA, so the
            // lit colour is colDown/cover; composite that over the frame by cov.
            for (int k = 0; k < 3; ++k) {
                float src = c[x][k] / cov;
                d[x][k] = cv::saturate_cast<uchar>(src * cov + d[x][k] * (1.f - cov));
            }
        }
    }
}

} // namespace olc::animal3d
