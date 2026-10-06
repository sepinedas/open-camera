#include "face3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace olc::face3d {

namespace {

using cv::Vec3f;
using cv::Matx33f;

constexpr float kPi = 3.14159265358979f;
// Supersampling for the silhouette. Ears and a nose cover far less of the
// frame than a whole face, so 2x2 is affordable and the edges need it.
constexpr int kSS = 2;
// Cap on supersampled pixels per frame.
//
// Supersampling costs its square, and it is paid over the whole region the
// model touches -- affordable for a pair of ears, ruinous for a shark's head,
// which covers ten times the area at the same distance. Rather than a fixed
// factor, spend the budget: a model small enough to fit inside it is drawn at
// the full kSS, and a larger one has its factor eased down so the frame costs
// the same whatever is on screen.
//
// Continuous rather than a choice between 1x and 2x, because a threshold puts
// a hump in the cost curve just under it and makes the edges visibly pop as
// someone leans toward the camera. At 420k, everything but the shark is at
// the full factor for any normal framing.
constexpr int kMaxSamples = 420000;

// --- Per-species geometry, in eye-separation units in the head frame -------
// +x toward the image-right eye, +y toward the chin, +z away from the camera,
// origin at the eye midpoint. Ear anchors are relative to the *measured* crown
// and head width, so they land on the real silhouette rather than an assumed
// one, and the muzzle sits on the measured nose.
struct Style {
    // Ear: a lobe swept from an attachment on the head out to a tip, built
    // as a closed shell so it has a front, a back and a visible thickness.
    float earAttachX, earAttachY;
    float earTipX, earTipY;
    float earHalfW;  // widest half-width, as a multiple of headHalfW
    float earBaseW;  // half-width where it meets the head, as a fraction of
                     // earHalfW. Too broad and the closing cap across the
                     // attachment hangs below the head as a flat dark panel.
    float earRound;  // tip shape: small rounds it off, large draws it to a point
    // The shell's cross-section. `earBowl` is signed and does most of the
    // work: positive hollows the front into a concha, which is what you see
    // on an ear that faces you, and negative domes it outward, which is what
    // you see on the *back* of an ear that hangs. A lobe with no bowl either
    // way is a horn, whatever outline it is given.
    float earBowl;
    float earRim;    // how far the rim stands proud of the bowl floor
    float earShell;  // front-to-back thickness, seen at the silhouette
    float earYaw;    // rotate the lobe about its own axis, so it faces outward
    float earCurl;   // forward curl toward the tip
    float earInnerAmt; // how strongly the inner colour floods the bowl
    float earMottle; // per-vertex coat variation, so it is not moulded plastic
    Vec3f earCol, earInnerCol;

    // Muzzle. A dog gets a bare domed nose sitting on the painted muzzle; a
    // pig gets a snout -- a short tube standing off the face, capped by a disc
    // with two nostrils -- which is the whole point of the animal.
    bool snout;
    float noseR;     // dog: radius of the nose dome
    float noseWide, noseTall; // dome aspect: a dog's is squat, a grinch's tall
    float noseYOff;  // slide the dome up the bridge, away from the lip
    // Dome shading. Wet leather is dark and glossy; skin is neither, and
    // lighting a skin-coloured nose like leather turns it into a dark blob.
    float noseAmbient, noseSpec;
    int noseShin;
    // Nostrils, carved into whichever surface the muzzle ends in -- the dome
    // for a dog or a grinch, the snout's end disc for a pig. Carved, not
    // pasted: a disc placed in front of the muzzle reads as a sticker, and
    // one placed behind it is simply hidden by the depth test.
    float nostrilX, nostrilY; // centre, as a fraction of the surface's radii
    float nostrilRx, nostrilRy;
    float nostrilDepth;
    float snoutLen;  // pig: how far the snout stands off the face
    float snoutR;    // pig: radius of the tube at the face
    float snoutFlat; // pig: disc height / width
    float snoutDrop; // pig: how far the axis tilts down as it comes forward
    Vec3f noseCol, snoutCol, nostrilCol;
    Vec3f tongueCol;
};

Style styleFor(Species sp) {
    Style s{};
    if (sp == Species::Dog) {
        s.earAttachX = 0.88f; s.earAttachY = 0.10f;
        s.earTipX = 0.99f;    s.earTipY = 1.32f;
        s.earHalfW = 0.31f;   s.earRound = 3.2f;
        s.earBaseW = 0.70f;
        // These ears hang, so what faces the camera is the *back* of the ear:
        // domed, not hollow. Hence a negative bowl, with only a hint of the
        // inner surface showing along the leading edge.
        s.earBowl = -0.15f;   s.earRim = 0.02f;
        s.earShell = 0.085f;  s.earYaw = 0.22f;
        s.earCurl = 0.12f;    s.earInnerAmt = 0.45f;
        s.earMottle = 0.07f;
        s.earCol = Vec3f(62, 96, 138);
        s.earInnerCol = Vec3f(78, 92, 126);
        s.snout = false;
        s.noseR = 0.24f;
        s.noseWide = 1.20f; s.noseTall = 0.88f; s.noseYOff = 0.f;
        s.noseAmbient = 0.30f; s.noseSpec = 0.55f; s.noseShin = 34;
        s.nostrilX = 0.46f; s.nostrilY = 0.30f;
        s.nostrilRx = 0.30f; s.nostrilRy = 0.42f;
        s.nostrilDepth = 0.30f;
        s.noseCol = Vec3f(30, 28, 28);
        s.nostrilCol = Vec3f(10, 9, 9);
        s.tongueCol = Vec3f(122, 108, 206);
    } else if (sp == Species::Grinch) {
        // Big pointed ears swept up and well out to the sides -- the most
        // far-reaching geometry of any species here, which is what sets the
        // dirty-region margin the animal filters need.
        // The attachment has to sit where the head is genuinely widest --
        // temple height, not the crown. Anchored up by the crown the lobes
        // leave a visible gap and read as leaves stuck on behind the head.
        s.earAttachX = 0.88f; s.earAttachY = 0.56f;
        s.earTipX = 1.38f;    s.earTipY = -0.44f;
        s.earHalfW = 0.40f;   s.earRound = 2.9f; // pointed, but not a spike
        s.earBaseW = 0.42f;
        // Upright and turned to face you, so the concha is the whole front of
        // the ear: a deep bowl inside a proud rim.
        s.earBowl = 0.20f;    s.earRim = 0.075f;
        s.earShell = 0.070f;  s.earYaw = 0.42f;
        s.earCurl = 0.05f;    s.earInnerAmt = 0.90f;
        s.earMottle = 0.10f;
        s.earCol = Vec3f(100, 205, 125);
        s.earInnerCol = Vec3f(58, 140, 84);
        s.snout = false;
        s.noseR = 0.235f;
        s.nostrilX = 0.42f; s.nostrilY = 0.44f;
        s.nostrilRx = 0.26f; s.nostrilRy = 0.34f;
        s.nostrilDepth = 0.26f;
        s.nostrilCol = Vec3f(46, 104, 62);
        s.tongueCol = Vec3f(112, 104, 192);
        // Taller than wide and slid up the bridge: an upturned snub, not the
        // squat leather pad a dog wears. Lit as skin, so it stays part of the
        // face rather than sitting on it as a dark bead.
        s.noseWide = 0.95f; s.noseTall = 1.30f; s.noseYOff = -0.06f;
        s.noseAmbient = 0.70f; s.noseSpec = 0.12f; s.noseShin = 16;
        s.noseCol = Vec3f(128, 218, 150);
    } else if (sp == Species::Elephant) {
        // Enormous, rounded, and hanging out and *down* rather than standing
        // up: an elephant's ear is nearly as big as its head, and getting
        // that scale right is most of what makes the filter read at a
        // glance. Barely hollowed -- what faces the camera is mostly the
        // flat of the ear.
        // NOTE the vertical numbers here are absolute eye separations, not
        // fractions of the head, so they have to be scaled up by hand for a
        // head this large -- left at animal values the ears come out as two
        // small flaps near the crown.
        s.earAttachX = 0.90f; s.earAttachY = -0.14f;
        s.earTipX = 1.60f;    s.earTipY = 1.34f;
        s.earHalfW = 0.76f;   s.earRound = 1.25f;
        s.earBaseW = 0.72f;
        s.earBowl = 0.10f;    s.earRim = 0.030f;
        s.earShell = 0.045f;  s.earYaw = 0.30f;
        s.earCurl = 0.10f;    s.earInnerAmt = 0.40f;
        s.earMottle = 0.06f;
        s.earCol = Vec3f(124, 125, 130);
        s.earInnerCol = Vec3f(146, 143, 148);
        s.snout = false;
        s.noseR = 0.12f;
        s.noseWide = 1.f; s.noseTall = 1.f; s.noseYOff = 0.f;
        s.noseAmbient = 0.40f; s.noseSpec = 0.20f; s.noseShin = 16;
        s.nostrilX = 0.42f; s.nostrilY = 0.f;
        s.nostrilRx = 0.28f; s.nostrilRy = 0.34f;
        s.nostrilDepth = 0.24f;
        s.noseCol = Vec3f(96, 96, 102);
        s.nostrilCol = Vec3f(52, 52, 58);
        s.tongueCol = Vec3f(120, 108, 198);
    } else if (sp == Species::Squirrel) {
        // Big rounded ears set high on the head and turned outward. Rounded
        // right off -- a pointed lobe up there reads as a horn, which is the
        // mistake the pig's ears started out making.
        s.earAttachX = 0.66f; s.earAttachY = 0.18f;
        s.earTipX = 0.84f;    s.earTipY = -0.60f;
        s.earHalfW = 0.46f;   s.earRound = 1.5f;
        s.earBaseW = 0.66f;
        s.earBowl = 0.17f;    s.earRim = 0.062f;
        s.earShell = 0.055f;  s.earYaw = 0.46f;
        s.earCurl = 0.09f;    s.earInnerAmt = 0.88f;
        s.earMottle = 0.09f;
        s.earCol = Vec3f(68, 112, 168);
        s.earInnerCol = Vec3f(104, 126, 176);
        s.snout = false;
        s.noseR = 0.15f;
        s.noseWide = 1.05f; s.noseTall = 0.82f; s.noseYOff = 0.02f;
        s.noseAmbient = 0.34f; s.noseSpec = 0.46f; s.noseShin = 28;
        s.nostrilX = 0.44f; s.nostrilY = 0.28f;
        s.nostrilRx = 0.28f; s.nostrilRy = 0.38f;
        s.nostrilDepth = 0.26f;
        s.noseCol = Vec3f(30, 30, 42);
        s.nostrilCol = Vec3f(12, 12, 18);
        s.tongueCol = Vec3f(124, 110, 202);
    } else if (sp == Species::Dragon) {
        // Not ears but frills: long pointed fins swept out and up from the
        // sides of the skull, a crimson rim round a gold membrane. Pointed
        // right to the tip -- on a dragon a horn-like silhouette is the point.
        s.earAttachX = 0.90f; s.earAttachY = 0.16f;
        s.earTipX = 1.95f;    s.earTipY = -0.62f;
        // A low earRound tapers the whole way to a point; a high one holds
        // the width to the end and then cuts it off, which reads as a rod.
        s.earHalfW = 0.27f;   s.earRound = 1.15f;
        s.earBaseW = 0.55f;
        s.earBowl = 0.12f;    s.earRim = 0.050f;
        s.earShell = 0.040f;  s.earYaw = 0.30f;
        s.earCurl = 0.10f;    s.earInnerAmt = 0.85f;
        s.earMottle = 0.06f;
        // Wide gold or orange inside reads as a mouse's ear whatever the
        // outline; a darker red membrane inside a crimson rim reads as a fin.
        s.earCol = Vec3f(36, 34, 160);
        s.earInnerCol = Vec3f(30, 70, 190);
        s.snout = false;
        s.noseR = 0.12f;
        s.noseWide = 1.f; s.noseTall = 1.f; s.noseYOff = 0.f;
        s.noseAmbient = 0.40f; s.noseSpec = 0.20f; s.noseShin = 16;
        s.nostrilX = 0.42f; s.nostrilY = 0.f;
        s.nostrilRx = 0.28f; s.nostrilRy = 0.34f;
        s.nostrilDepth = 0.24f;
        s.noseCol = Vec3f(40, 38, 170);
        s.nostrilCol = Vec3f(20, 20, 60);
        s.tongueCol = Vec3f(110, 90, 210);
    } else { // Pig
        // Ears stand up off the crown and lean outward: the tip's y is
        // negative, i.e. *above* the crown, which is what keeps them clear of
        // the face entirely.
        s.earAttachX = 0.56f; s.earAttachY = 0.12f;
        s.earTipX = 0.86f;    s.earTipY = -0.80f;
        // Broad and blunt. A narrow pointed lobe on top of a head is a horn
        // no matter how it is shaded, so the width goes up and the tip is
        // rounded right off.
        s.earHalfW = 0.50f;   s.earRound = 1.7f;
        s.earBaseW = 0.72f;
        s.earBowl = 0.22f;    s.earRim = 0.070f;
        s.earShell = 0.060f;  s.earYaw = 0.50f;
        s.earCurl = 0.16f;    s.earInnerAmt = 0.95f;
        s.earMottle = 0.05f;
        s.earCol = Vec3f(172, 152, 239);
        s.earInnerCol = Vec3f(118, 92, 198);
        s.snout = true;
        s.snoutLen = 0.34f;
        s.snoutR = 0.46f;
        s.snoutFlat = 0.80f;
        s.snoutDrop = 0.16f;
        s.nostrilX = 0.40f; s.nostrilY = 0.00f;
        s.nostrilRx = 0.23f; s.nostrilRy = 0.34f;
        s.nostrilDepth = 0.23f;
        s.snoutCol = Vec3f(178, 160, 242);
        s.nostrilCol = Vec3f(74, 54, 122);
        s.tongueCol = Vec3f(134, 116, 214);
    }
    return s;
}

float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// A little per-vertex colour variation, so the ears do not read as moulded
// plastic beside the furred face. The ear grid is only 13x14, so this is a
// soft mottle rather than the fine strokes painted onto the mesh -- which is
// about right for an ear anyway, where the fur is shorter and flatter.
float mottle(int a, int b) {
    // Unsigned multiply: the signed form overflows and is undefined.
    uint32_t h = (uint32_t)a * 374761393u ^ (uint32_t)b * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0xFFFFFFu) * (1.f / (float)0xFFFFFF) * 2.f - 1.f;
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
    int shin = 14; // specular exponent; integer, for fastPow

    // How the face mesh moves it (see deform()). `jaw` picks which half of
    // the face it follows. `bind` is where each vertex reads its motion from;
    // empty, or shorter than `pos`, means from where the vertex itself is.
    // A part that should ride the surface without stretching -- an eye, a
    // tooth, a horn -- binds all its vertices to one point, so it moves as a
    // piece with whatever it is set into.
    bool jaw = false;
    std::vector<Vec3f> bind;

    int add(const Vec3f& p, const Vec3f& c) {
        pos.push_back(p);
        col.push_back(c);
        nrm.emplace_back(0.f, 0.f, 0.f);
        return (int)pos.size() - 1;
    }
    void face(int a, int b, int c) { tri.emplace_back(a, b, c); }

    // Bind every vertex from `from` on to `at`. Vertices before it that have
    // no binding yet keep their own position.
    void pin(int from, const Vec3f& at) {
        for (size_t i = bind.size(); i < pos.size(); ++i) bind.push_back(pos[i]);
        for (size_t i = (size_t)from; i < pos.size(); ++i) bind[i] = at;
    }

    // Make every triangle wind outward, for a mesh that is star-shaped about
    // its own centroid.
    //
    // Winding by hand has gone wrong four times here, and it fails silently:
    // a disc built as a fan winds opposite to the tube it closes, back-face
    // culling drops the whole disc, and the pig is left with a hollow ring
    // where its snout should be. Inconsistent winding also makes
    // computeNormals() cancel neighbouring faces against each other, so the
    // shading goes wrong even where culling is off.
    //
    // For a shell with no concavities the outward direction can simply be
    // measured instead of reasoned about, which is what this does. It is NOT
    // valid once a surface has pits cut into it -- inside a nostril the wall
    // faces back toward the centroid and would be flipped inside out -- so
    // the carved surfaces are built on a single uniform grid instead, where
    // one winding pattern is correct everywhere by construction.
    void orientOutward() {
        if (pos.empty()) return;
        Vec3f c(0.f, 0.f, 0.f);
        for (const auto& p : pos) c += p;
        c = c * (1.f / (float)pos.size());
        for (auto& t : tri) {
            const Vec3f& a = pos[t[0]];
            const Vec3f fn = (pos[t[1]] - a).cross(pos[t[2]] - a);
            const Vec3f out = (a + pos[t[1]] + pos[t[2]]) * (1.f / 3.f) - c;
            if (fn.dot(out) < 0.f) std::swap(t[1], t[2]);
        }
    }

    // Draw near geometry first, so the depth test can reject what is behind
    // it before any of it is shaded.
    //
    // The lofts are built back-to-front -- station 0 is the back of the head
    // -- which is the worst possible order for a z-buffer: every hidden pixel
    // gets lit, written, and then painted over by the near geometry that
    // follows. The depth test already rejects before shading; it just never
    // got the chance. This reorders the triangle list and nothing else, so
    // the image is unchanged and only the work to reach it differs.
    void sortNearFirst() {
        std::sort(tri.begin(), tri.end(),
                  [&](const cv::Vec3i& a, const cv::Vec3i& b) {
                      const float za = pos[a[0]][2] + pos[a[1]][2] + pos[a[2]][2];
                      const float zb = pos[b[0]][2] + pos[b[1]][2] + pos[b[2]][2];
                      return za < zb; // -z is toward the camera
                  });
    }

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

// Rotation about an arbitrary unit axis (Rodrigues). The ears need it: their
// sweep axis is not one of the head's, and they have to be rolled about it so
// the bowl faces outward rather than straight at the camera.
Matx33f rotAxis(const Vec3f& u, float a) {
    const float c = std::cos(a), sn = std::sin(a), k = 1.f - c;
    const float x = u[0], y = u[1], z = u[2];
    return Matx33f(c + x * x * k,     x * y * k - z * sn, x * z * k + y * sn,
                   y * x * k + z * sn, c + y * y * k,     y * z * k - x * sn,
                   z * x * k - y * sn, z * y * k + x * sn, c + z * z * k);
}

Matx33f rotZ(float a) {
    float c = std::cos(a), s = std::sin(a);
    return Matx33f(c, -s, 0, s, c, 0, 0, 0, 1);
}

// --- Mesh builders ---------------------------------------------------------

Mesh buildEar(float side, float wiggle, const Head& h, const Style& st) {
    Mesh m;
    m.doubleSided = true; // a closed shell, but never rely on winding
    m.ambient = 0.38f;
    m.spec = 0.12f;
    m.shin = 12;

    // An ear is a lobe swept along an axis from where it joins the head to its
    // tip, widening out of the attachment and closing again at the far end.
    // Sweeping a *width profile* is what makes it read as an ear rather than a
    // flat triangle; giving that sheet a hollow, a rim and a back is what
    // stops the result reading as a horn.
    //
    // Model eyes are at (+-0.5, -0.35); +y is down, -z is toward the camera.
    const float hw = h.headHalfW;
    const float cy = h.crownY;
    const Vec3f attach(side * st.earAttachX * hw, cy + st.earAttachY, -0.05f);
    const Vec3f tip(side * st.earTipX * hw, cy + st.earTipY, -0.20f);
    const Vec3f axis = tip - attach;
    const Vec3f uAxis = norm(axis);
    // Across the lobe, perpendicular to its axis and to the view direction, so
    // the width starts out spread across the screen rather than into it.
    Vec3f across = norm(Vec3f(0.f, 0.f, -1.f).cross(uAxis));
    if (across[0] * side < 0.f) across = -across; // keep +across pointing outward
    Vec3f Nrm = norm(across.cross(uAxis));
    if (Nrm[2] > 0.f) Nrm = -Nrm; // face the camera
    // Roll the whole lobe about its own axis so the bowl looks outward and
    // forward, the way an ear sits on a head, instead of straight down the
    // lens like a satellite dish.
    {
        const Matx33f Ry = rotAxis(uAxis, -side * st.earYaw);
        across = Ry * across;
        Nrm = Ry * Nrm;
    }
    const float maxHalfW = st.earHalfW * hw;

    // Half-width along the lobe: broad where it meets the head, widest a third
    // of the way along, then closing. `earRound` sets how abruptly it closes --
    // a small value rounds the tip off, a large one draws it to a point.
    auto halfWidth = [&](float t) {
        const float base = st.earBaseW;
        const float body = base + (1.f - base) *
                           std::sin(kPi * clampf(t * 0.85f + 0.08f, 0.f, 1.f));
        const float close = std::pow(std::max(0.f, 1.f - std::pow(t, st.earRound)), 0.45f);
        return maxHalfW * body * close;
    };
    // How much hollow there is at each point along the lobe: shallow where it
    // leaves the head, deepest a little under half way up, fading at the tip.
    auto bowlAt = [&](float t) {
        return std::sin(kPi * clampf(0.20f + 0.70f * t, 0.f, 1.f));
    };

    const int nA = 15, nT = 16;
    std::vector<std::vector<int>> F(nT, std::vector<int>(nA));
    std::vector<std::vector<int>> B(nT, std::vector<int>(nA));
    for (int ti = 0; ti < nT; ++ti) {
        const float t = (float)ti / (nT - 1); // 0 at the head, 1 at the tip
        const Vec3f centre = attach + axis * t;
        const float w = halfWidth(t);
        const float bowl = bowlAt(t);
        for (int ai = 0; ai < nA; ++ai) {
            const float a = -1.f + 2.f * (float)ai / (nA - 1); // -1..1 across
            const float e = std::fabs(a);                      // 0 centre, 1 rim
            const Vec3f mid = centre + across * (w * a);

            // Front surface: the rim rises steeply at the edge, the middle is
            // pushed away from the camera into the concha. A negative bowl
            // domes it instead, for an ear whose back is what faces you.
            const float dF = st.earRim * std::pow(e, 2.4f)
                           - st.earBowl * (1.f - e * e) * bowl
                           + st.earCurl * t * t;
            // The back always sits behind the front by a positive amount, so
            // the shell can never turn itself inside out however the bowl is
            // signed. Thicker down the middle, thinning to the rim.
            const float dB = dF - st.earShell * (0.55f + 0.45f * (1.f - e * e));

            const float mot = 1.f + st.earMottle * mottle(ti, ai * 7 + (int)side);
            // The bowl floods with the inner colour; the rim keeps the coat's.
            const float inF = st.earInnerAmt *
                              clampf(1.35f * (1.f - e), 0.f, 1.f) * bowl;
            const Vec3f inner = (st.earCol * (1.f - inF) + st.earInnerCol * inF) * mot;
            // Rim highlight, and a back that sits in its own shadow.
            const Vec3f outer = st.earCol * (1.f + 0.06f * e) * mot;
            const Vec3f backCol = st.earCol * 0.80f * mot;

            F[ti][ai] = m.add(mid + Nrm * dF, inner * (1.f - e * e) + outer * (e * e));
            B[ti][ai] = m.add(mid + Nrm * dB, backCol);
        }
    }

    for (int ti = 0; ti + 1 < nT; ++ti)
        for (int ai = 0; ai + 1 < nA; ++ai) {
            m.face(F[ti][ai], F[ti][ai + 1], F[ti + 1][ai + 1]);
            m.face(F[ti][ai], F[ti + 1][ai + 1], F[ti + 1][ai]);
            // Reversed, so the back sheet winds outward like the front.
            m.face(B[ti][ai], B[ti + 1][ai + 1], B[ti][ai + 1]);
            m.face(B[ti][ai], B[ti + 1][ai], B[ti + 1][ai + 1]);
        }
    // Side walls down both edges, and a cap where the ear meets the head, so
    // the shell is closed and shows a real thickness at the silhouette.
    for (int ti = 0; ti + 1 < nT; ++ti)
        for (int ai : {0, nA - 1}) {
            m.face(F[ti][ai], B[ti][ai], B[ti + 1][ai]);
            m.face(F[ti][ai], B[ti + 1][ai], F[ti + 1][ai]);
        }
    for (int ai = 0; ai + 1 < nA; ++ai) {
        m.face(F[0][ai], B[0][ai], B[0][ai + 1]);
        m.face(F[0][ai], B[0][ai + 1], F[0][ai + 1]);
    }

    // Idle wiggle: rock the whole ear about its attachment in the image plane.
    if (wiggle != 0.f) {
        Matx33f Rw = rotZ(side * wiggle);
        Vec3f pivot = attach;
        for (auto& p : m.pos) p = rotAbout(Rw, p, pivot);
    }
    // Rides the head where it joins it, as a piece: stretched by the face's
    // motion an ear would bend wherever the field happened to fade.
    m.pin(0, attach);
    m.orientOutward();
    m.computeNormals();
    return m;
}

// The nose: a glossy dome standing off the face at the measured nose tip.
// Wound so its outward face is the front face -- copying the winding of a
// forward-sweeping tube instead leaves it entirely back-facing, and culling
// swallows all but a sliver of rim.
// How deeply a nostril is cut at a point on a muzzle's end surface. `(px, py)`
// are that point's coordinates on the surface, normalised so its rim is at
// radius 1; the two nostrils are mirrored about x. Returns 0..1.
float nostrilCut(float px, float py, const Style& st) {
    float cut = 0.f;
    for (float side : {-1.f, 1.f}) {
        const float dx = (px - side * st.nostrilX) / st.nostrilRx;
        const float dy = (py - st.nostrilY) / st.nostrilRy;
        const float d = std::sqrt(dx * dx + dy * dy);
        // A rounded pit with a soft lip, rather than a cylinder punched in.
        cut = std::max(cut, clampf(1.f - d * d, 0.f, 1.f));
    }
    return cut;
}

Mesh buildNose(const Head& h, const Style& st) {
    Mesh m;
    m.doubleSided = true; // carved: a nostril's near wall is a back face
    m.ambient = st.noseAmbient;
    m.spec = st.noseSpec;
    m.shin = st.noseShin;

    const Vec3f centre(0.f, h.noseY + st.noseYOff, h.noseZ);
    const int nSeg = 30, nRing = 11;
    std::vector<std::vector<int>> ring(nRing);
    for (int r = 0; r < nRing; ++r) {
        // 0 at the equator, against the face; 1 at the pole, toward the camera.
        const float lat = 0.5f * kPi * (float)r / (nRing - 1);
        const float cl = std::cos(lat), sl = std::sin(lat);
        for (int i = 0; i < nSeg; ++i) {
            const float a = 2.f * kPi * i / nSeg;
            const float ca = std::cos(a), sa = std::sin(a);
            // The unit sphere this dome is a scaled copy of; also its normal
            // direction, which is the direction a nostril is cut along.
            const Vec3f u(cl * ca, cl * sa, -sl);
            // Aspect comes from the style: leather is wider than tall, a
            // grinch's nose is the other way about.
            Vec3f p = centre + Vec3f(st.noseWide * st.noseR * u[0],
                                     st.noseTall * st.noseR * u[1],
                                     st.noseR * u[2]);
            // Nostrils, cut into the front of the dome along its own normal so
            // they are holes in the muzzle rather than beads stuck on it. The
            // cut is measured across the dome's face -- the projected (x, y) --
            // so it stays put as the head turns.
            const float cut = nostrilCut(cl * ca, cl * sa, st) * sl;
            if (cut > 0.f) p += u * (st.nostrilDepth * st.noseR * cut);
            const Vec3f col = st.noseCol * (1.f - cut) + st.nostrilCol * cut;
            ring[r].push_back(m.add(p, col));
        }
    }
    for (int r = 0; r + 1 < nRing; ++r)
        for (int i = 0; i < nSeg; ++i) {
            const int j = (i + 1) % nSeg;
            m.face(ring[r][i], ring[r][j], ring[r + 1][j]);
            m.face(ring[r][i], ring[r + 1][j], ring[r + 1][i]);
        }
    m.computeNormals();
    return m;
}


// --- Pig snout -------------------------------------------------------------
// A short elliptical tube standing off the nose, capped by a domed disc
// with two nostrils cut into it.
struct SnoutFrame {
    Vec3f base, axis, u, v, pad;
    float len, rx, ry;
};

SnoutFrame snoutFrame(const Head& h, const Style& st) {
    SnoutFrame s;
    s.axis = norm(Vec3f(0.f, st.snoutDrop, -1.f)); // forward, a touch down
    s.len = st.snoutLen;
    s.rx = st.snoutR;
    s.ry = st.snoutFlat * s.rx;
    // Sits on the measured nose and protrudes forward from there.
    s.base = Vec3f(0.f, h.noseY, h.noseZ);
    s.pad = s.base + s.axis * s.len;
    // An orthonormal pair spanning the disc: +u roughly head-right.
    Vec3f up(0.f, 1.f, 0.f);
    if (std::fabs(s.axis.dot(up)) > 0.9f) up = Vec3f(1.f, 0.f, 0.f);
    s.u = norm(up.cross(s.axis));
    s.v = norm(s.axis.cross(s.u));
    return s;
}

Mesh buildSnout(const Head& h, const Style& st) {
    Mesh m;
    // Carved, so parts of it face away from the camera and have to be drawn
    // and depth-tested rather than culled: the near wall of a nostril is a
    // back face, and culling it would leave a hole straight through the snout.
    m.doubleSided = true;
    m.ambient = 0.42f;
    m.spec = 0.17f; // skin, not lacquer
    m.shin = 18;
    const SnoutFrame sf = snoutFrame(h, st);
    const int nSeg = 34, nTube = 5, nPad = 8;

    // One grid, swept in a single direction: up the tube from the face, out
    // over the rolled rim, then in across the end disc to its centre. Because
    // it is one grid, one triangle pattern winds the whole thing consistently
    // -- which is the bit that keeps going wrong when the disc is built as a
    // separate fan and stitched on.
    std::vector<std::vector<int>> ring;
    ring.reserve(nTube + nPad);

    for (int r = 0; r < nTube; ++r) {
        const float t = (float)r / (nTube - 1);
        const Vec3f c = sf.base + sf.axis * (t * sf.len);
        const float rx = sf.rx * (1.f + 0.14f * t); // flares toward the disc
        const float ry = sf.ry * (1.f + 0.12f * t);
        ring.emplace_back();
        for (int i2 = 0; i2 < nSeg; ++i2) {
            const float a = 2.f * kPi * i2 / nSeg;
            ring.back().push_back(m.add(c + sf.u * (rx * std::cos(a)) +
                                            sf.v * (ry * std::sin(a)),
                                        st.snoutCol));
        }
    }

    // The end disc: a rolled lip around a domed face, with the nostrils cut
    // into it. Concentric rings from the rim inward, so there are vertices to
    // carve -- a single fan has none.
    const Vec3f padCol = st.snoutCol * 1.08f;
    const float frx = sf.rx * 1.16f, fry = sf.ry * 1.14f;
    for (int r = 0; r < nPad; ++r) {
        const float q = 1.f - (float)r / (nPad - 1); // 1 at the rim, 0 centre
        ring.emplace_back();
        for (int i2 = 0; i2 < nSeg; ++i2) {
            const float a = 2.f * kPi * i2 / nSeg;
            const float px = q * std::cos(a), py = q * std::sin(a);
            // Domed in the middle and rolling back at the very edge, so the
            // rim reads as a lip rather than a cut-off cylinder.
            float lift = 0.075f * (1.f - q * q) - 0.045f * std::pow(q, 6.f);
            const float cut = nostrilCut(px, py, st);
            lift -= st.nostrilDepth * cut;
            const Vec3f col = padCol * (1.f - cut) + st.nostrilCol * cut;
            ring.back().push_back(m.add(sf.pad + sf.u * (frx * px) +
                                            sf.v * (fry * py) + sf.axis * lift,
                                        col));
        }
    }

    for (size_t r = 0; r + 1 < ring.size(); ++r)
        for (int i2 = 0; i2 < nSeg; ++i2) {
            const int j2 = (i2 + 1) % nSeg;
            m.face(ring[r][i2], ring[r][j2], ring[r + 1][j2]);
            m.face(ring[r][i2], ring[r + 1][j2], ring[r + 1][i2]);
        }
    m.computeNormals();
    return m;
}


// A rounded lobe: half an ellipsoid facing the camera. The squirrel's cheek
// pouches are these, and so is a nose once the nostrils are cut into it --
// this is the plain version, with nothing carved.
Mesh buildLobe(const Vec3f& centre, float rx, float ry, float rz,
               const Vec3f& col, float ambient, float spec, int shin) {
    Mesh m;
    m.doubleSided = true;
    m.ambient = ambient;
    m.spec = spec;
    m.shin = shin;
    const int nSeg = 20, nRing = 7;
    std::vector<std::vector<int>> ring(nRing);
    for (int r = 0; r < nRing; ++r) {
        const float lat = 0.5f * kPi * (float)r / (nRing - 1);
        const float cl = std::cos(lat), sl = std::sin(lat);
        for (int i = 0; i < nSeg; ++i) {
            const float a = 2.f * kPi * i / nSeg;
            ring[r].push_back(m.add(centre + Vec3f(rx * cl * std::cos(a),
                                                   ry * cl * std::sin(a),
                                                   -rz * sl),
                                    col));
        }
    }
    for (int r = 0; r + 1 < nRing; ++r)
        for (int i = 0; i < nSeg; ++i) {
            const int j = (i + 1) % nSeg;
            m.face(ring[r][i], ring[r][j], ring[r + 1][j]);
            m.face(ring[r][i], ring[r + 1][j], ring[r + 1][i]);
        }
    m.pin(0, centre);
    m.computeNormals();
    return m;
}

// A pair of incisors: flat blades hanging from `top` down the front of the
// muzzle, rounded off at the bottom. They are always showing a little -- that
// is the point of a squirrel -- and come further down as the jaw opens.
//
// Placed explicitly rather than derived from the Head: where the front of a
// muzzle is depends entirely on the model wearing it, and offsets that suited
// a face with paint on it buried these inside a lofted head.
Mesh buildIncisors(const Vec3f& top, float len, float halfW, float thick,
                   const Vec3f& col) {
    Mesh m;
    m.doubleSided = true;
    m.ambient = 0.62f;
    m.spec = 0.26f;
    m.shin = 24.f;
    const int nT = 7, nA = 7;
    for (float side : {-1.f, 1.f}) {
        const float cx = top[0] + side * (halfW + 0.012f);
        const float topY = top[1];
        const float z0 = top[2];
        std::vector<std::vector<int>> front(nT), back(nT);
        for (int ti = 0; ti < nT; ++ti) {
            const float t = (float)ti / (nT - 1);
            // Rounded at the tip, so it reads as a tooth and not a peg.
            const float w = halfW * std::sqrt(std::max(0.f, 1.f - t * t * t));
            const float y = topY + len * t;
            for (int ai = 0; ai < nA; ++ai) {
                const float a = -1.f + 2.f * (float)ai / (nA - 1);
                const float dome = std::sqrt(std::max(0.f, 1.f - a * a));
                const Vec3f mid(cx + w * a, y, z0);
                front[ti].push_back(m.add(mid + Vec3f(0.f, 0.f, -thick * dome),
                                          col * (0.93f + 0.09f * dome)));
                back[ti].push_back(m.add(mid + Vec3f(0.f, 0.f, thick * 0.4f),
                                         col * 0.70f));
            }
        }
        for (int ti = 0; ti + 1 < nT; ++ti)
            for (int ai = 0; ai + 1 < nA; ++ai) {
                m.face(front[ti][ai], front[ti][ai + 1], front[ti + 1][ai + 1]);
                m.face(front[ti][ai], front[ti + 1][ai + 1], front[ti + 1][ai]);
                m.face(back[ti][ai], back[ti + 1][ai + 1], back[ti][ai + 1]);
                m.face(back[ti][ai], back[ti + 1][ai], back[ti + 1][ai + 1]);
            }
        for (int ti = 0; ti + 1 < nT; ++ti)
            for (int ai : {0, nA - 1}) {
                m.face(front[ti][ai], back[ti][ai], back[ti + 1][ai]);
                m.face(front[ti][ai], back[ti + 1][ai], front[ti + 1][ai]);
            }
    }
    m.pin(0, top);
    m.computeNormals();
    return m;
}

// A tube that tapers along a curving path: the elephant's trunk, and each of
// its tusks. One builder, because they differ only in how far they reach, how
// hard they curl and what colour they are.
//
// The path is integrated rather than written down: at each step the direction
// turns from `a0` toward `a1` -- angles measured from straight down toward the
// camera -- so the thing bends continuously instead of being a polyline with
// corners in it. Past a quarter turn the tip is rising, which is what lets the
// trunk curl up.
Mesh buildTaperTube(const Vec3f& base, float a0, float a1, float sideX,
                    float length, float rBase, float rTipFrac,
                    const Vec3f& col, float band, float ambient, float spec,
                    int shin) {
    Mesh m;
    m.doubleSided = true; // it curls back on itself; winding is not worth it
    m.ambient = ambient;
    m.spec = spec;
    m.shin = shin;

    const int nS = 20, nSeg = 14;
    const float step = length / (nS - 1);
    Vec3f pos = base;
    std::vector<std::vector<int>> ring(nS);
    for (int si = 0; si < nS; ++si) {
        const float t = (float)si / (nS - 1);
        // Turns harder toward the tip, so the curl gathers at the end.
        const float a = a0 + (a1 - a0) * t * t;
        const Vec3f dir = norm(Vec3f(sideX, std::cos(a), -std::sin(a)));
        // A frame across the tube. `up` is kept out of the turning plane so
        // the cross-sections do not spin as the path bends.
        Vec3f u = norm(Vec3f(1.f, 0.f, 0.f).cross(dir));
        if (std::fabs(u.dot(u)) < 1e-6f) u = Vec3f(0.f, 0.f, 1.f);
        const Vec3f v = norm(dir.cross(u));
        const float r = rBase * (1.f - (1.f - rTipFrac) * std::pow(t, 0.85f));
        // Ringed, the way a trunk is: a gentle banding along its length.
        const float ribbed = 1.f + band * std::sin(t * 46.f);
        for (int i = 0; i < nSeg; ++i) {
            const float ang = 2.f * kPi * i / nSeg;
            ring[si].push_back(m.add(pos + u * (r * std::cos(ang)) +
                                         v * (r * std::sin(ang)),
                                     col * ribbed));
        }
        pos += dir * step;
    }
    for (int si = 0; si + 1 < nS; ++si)
        for (int i = 0; i < nSeg; ++i) {
            const int j = (i + 1) % nSeg;
            m.face(ring[si][i], ring[si][j], ring[si + 1][j]);
            m.face(ring[si][i], ring[si + 1][j], ring[si + 1][i]);
        }
    // Cap the tip, so a curled trunk does not show a hole down its end.
    {
        Vec3f mid(0.f, 0.f, 0.f);
        for (int i = 0; i < nSeg; ++i) mid += m.pos[ring[nS - 1][i]];
        const int c0 = m.add(mid * (1.f / (float)nSeg), col * 0.9f);
        for (int i = 0; i < nSeg; ++i)
            m.face(c0, ring[nS - 1][i], ring[nS - 1][(i + 1) % nSeg]);
    }
    m.pin(0, base);
    m.computeNormals();
    return m;
}

// A cone with an elliptical base, from `base` along `dir` to a point: the
// dinosaur's spikes and horns. `rA` is the base radius across `dir` in the
// head's x, `rB` the other way, so a spike can be a thin plate rather than a
// round peg. The base is sunk a little below where it is placed, so the cone
// grows out of the surface instead of perching on it.
//
// Leaned back on itself as it rises (`sweep`, along `back`), so it reads as a
// spike rather than a traffic cone. Coloured from `col` at the root to `tipCol`
// at the point.
Mesh buildCone(const Vec3f& base, const Vec3f& dir, const Vec3f& back,
               float height, float rA, float rB, float sweep,
               const Vec3f& col, const Vec3f& tipCol) {
    Mesh m;
    m.doubleSided = true;
    m.ambient = 0.46f;
    m.spec = 0.22f;
    m.shin = 18;
    const Vec3f d = norm(dir);
    Vec3f a = norm(Vec3f(1.f, 0.f, 0.f) - d * d[0]);
    if (a.dot(a) < 1e-6f) a = Vec3f(0.f, 0.f, 1.f);
    const Vec3f b = norm(d.cross(a));
    const int nSeg = 14, nT = 6;
    std::vector<std::vector<int>> ring(nT);
    for (int ti = 0; ti < nT; ++ti) {
        const float t = (float)ti / (nT - 1) * 0.92f; // stop short; apex below
        const float r = 1.f - t;
        const Vec3f c = base + d * (height * (t - 0.12f)) +
                        back * (sweep * height * t * t);
        const Vec3f cc = col * (1.f - t) + tipCol * t;
        for (int i = 0; i < nSeg; ++i) {
            const float ang = 2.f * kPi * i / nSeg;
            ring[ti].push_back(m.add(c + a * (rA * r * std::cos(ang)) +
                                         b * (rB * r * std::sin(ang)),
                                     cc));
        }
    }
    for (int ti = 0; ti + 1 < nT; ++ti)
        for (int i = 0; i < nSeg; ++i) {
            const int j = (i + 1) % nSeg;
            m.face(ring[ti][i], ring[ti][j], ring[ti + 1][j]);
            m.face(ring[ti][i], ring[ti + 1][j], ring[ti + 1][i]);
        }
    const int apex = m.add(base + d * (height * 0.88f) + back * (sweep * height),
                           tipCol);
    for (int i = 0; i < nSeg; ++i)
        m.face(ring[nT - 1][i], ring[nT - 1][(i + 1) % nSeg], apex);
    m.pin(0, base);
    m.computeNormals();
    return m;
}

// A tongue: a flattened slab swept from `base` toward `tip`, rounded off at
// the end and given a little thickness, so it reads from the side as well as
// head-on. Shared -- a shark's lolls out of its jaw, a dog's out of its
// muzzle, and only the placement differs.
Mesh buildTongue(const Vec3f& base, const Vec3f& tip, float halfW,
                 float thick, float curl, const Vec3f& col) {
    Mesh m;
    m.doubleSided = true; // small, and not worth reasoning about winding for
    m.ambient = 0.52f;
    m.spec = 0.34f;
    m.shin = 20.f;

    const Vec3f axis = tip - base;
    const float L = std::sqrt(axis.dot(axis));
    if (L < 1e-3f) return m;
    const Vec3f u = axis * (1.f / L);
    // Across the tongue, kept spread on screen rather than into it.
    Vec3f across = norm(Vec3f(0.f, 0.f, -1.f).cross(u));
    if (std::fabs(across.dot(across)) < 1e-6f) across = Vec3f(1.f, 0.f, 0.f);
    const Vec3f up = norm(across.cross(u));

    const int nT = 12, nA = 9;
    std::vector<std::vector<int>> top(nT), bot(nT);
    for (int ti = 0; ti < nT; ++ti) {
        const float t = (float)ti / (nT - 1);
        // Broad at the root, rounding to a blunt point.
        const float w = halfW * std::sqrt(std::max(0.f, 1.f - t * t * t * 0.92f));
        // Curls toward the end, as a lolling tongue does. Which way depends
        // on where it is hanging from, so the caller decides the sign.
        const Vec3f c = base + u * (L * t) + up * (curl * L * t * t);
        const float th = thick * (1.f - 0.45f * t);
        // Darker down the centre groove, lighter at the edges.
        for (int ai = 0; ai < nA; ++ai) {
            const float a = -1.f + 2.f * (float)ai / (nA - 1);
            const float dome = std::sqrt(std::max(0.f, 1.f - a * a));
            const Vec3f mid = c + across * (w * a);
            const Vec3f cc = col * (0.86f + 0.20f * std::fabs(a));
            top[ti].push_back(m.add(mid - up * (th * dome), cc));
            bot[ti].push_back(m.add(mid + up * (th * dome * 0.65f), col * 0.72f));
        }
    }
    for (int ti = 0; ti + 1 < nT; ++ti)
        for (int ai = 0; ai + 1 < nA; ++ai) {
            m.face(top[ti][ai], top[ti][ai + 1], top[ti + 1][ai + 1]);
            m.face(top[ti][ai], top[ti + 1][ai + 1], top[ti + 1][ai]);
            m.face(bot[ti][ai], bot[ti + 1][ai + 1], bot[ti][ai + 1]);
            m.face(bot[ti][ai], bot[ti + 1][ai], bot[ti + 1][ai + 1]);
        }
    for (int ti = 0; ti + 1 < nT; ++ti)
        for (int ai : {0, nA - 1}) {
            m.face(top[ti][ai], bot[ti][ai], bot[ti + 1][ai]);
            m.face(top[ti][ai], bot[ti + 1][ai], top[ti + 1][ai]);
        }
    m.pin(0, base);
    m.computeNormals();
    return m;
}

// --- Shark -----------------------------------------------------------------
// Not a set of parts hung over a painted face: a whole head, built to the
// measured size of the real one and lofted along its own axis. Everything is
// derived from the head's own dimensions -- crown, chin, temples, nose -- so
// it fits the face it is worn by rather than an assumed one.
//
// The head is two lofts sharing one profile: the skull and a jaw that hinges
// against it. Where they meet is the mouth line, and because the loft's
// cross-section is an ellipse centred on that line, the line needs no separate
// definition -- it is simply y = cy(s), the centre of each cross-section.
// The silhouette is sampled at this many control points, evenly spaced along
// the head, for the species whose outline is given that way.
constexpr int kProfN = 9;

struct Section {
    float z, cy, rx, ryUp, ryLo, gape;
};

bool mapped(const Head& h);
struct HeadShape;
Section sectionAt(float s, const HeadShape& p);
Section coneSection(float s, const HeadShape& p);   // the shark's outline
Section muzzleSection(float s, const HeadShape& p); // the squirrel's
Section domedSection(float s, const HeadShape& p);  // the dinosaur's

// A whole head, in place of the one that is there: a skull and a jaw that
// hinges against it, swept along the head's own longitudinal axis from s = 0
// at the back of the skull to s = 1 at the point of the muzzle.
//
// The silhouette is given as three curves rather than one axis plus a radius,
// because head-on the outline is what has to be got right and an
// axis-plus-radius body cannot produce it: whatever tapers forward hides
// inside the largest cross-section, which is the one that has to cover the
// whole head. So the muzzle is driven *past* the chin instead, where it is
// actually seen.
//
// Everything species-specific lives here, so the loft below does not care
// whether it is building a shark or a squirrel.
struct HeadShape {
    float crown, chin;     // the real head, which everything is sized to
    float topBack;         // top of the skull, a little clear of the crown
    float cyBack, cyTip;   // the mouth line, at the corners and at the point
    float botBack;         // bottom of the jaw, at the back
    float rxMax;
    float sHinge;          // where the jaw parts company with the skull
    float zBack, zTip;

    // --- how the silhouette is described ---------------------------------
    //
    // Two kinds of head need two kinds of description, so the shape carries
    // the one it uses and a pointer to the code that reads it.
    //
    // A cone (the shark) is three power curves and a monotonic taper: girth =
    // (1 - s^girthPow)^girthRoot, times a gentle bulge.
    float girthPow, girthRoot, girthBulge;
    float topPow, cyPow, botPow;
    // A head with a muzzle on it (the squirrel) is not any exponent: it is
    // wide and round over the cranium, then steps in. Those are control
    // points, sampled evenly from s = 0 at the back to s = 1 at the point --
    // `top`, `mouth` and `bot` in fractions of the head's length below the
    // crown, `girth` as a fraction of rxMax.
    float top[kProfN], mouth[kProfN], bot[kProfN], girth[kProfN];

    Section (*sectionFn)(float s, const HeadShape& p);

    // Countershading, keyed to height down the model. `waterA` is where the
    // back gives way to the flank and `waterB` where the flank gives way to
    // the belly, both as a fraction of the model's own depth.
    Vec3f back, flank, belly;
    float waterA, waterB;
    Vec3f gullet;          // the inside of the mouth
    float fur;             // per-vertex break-up: 0 for skin, more for fur
    bool gills;
    // Dark bands across the back, 0 for none. The dinosaur's: on a plain
    // green hide the loft reads as a smooth toy, and banding is what turns it
    // into a reptile's skin.
    float stripes;

    Expression e;          // what the face wearing it is doing
};

HeadShape sharkShape(const Head& h) {
    // The mouth line's smile and sulk, when the mesh is not there to move it
    // point by point (see MotionMap). With it, these would count it twice.
    const float smile = mapped(h) ? 0.f : h.expr.smile;
    const float sad = mapped(h) ? 0.f : h.expr.sad;
    HeadShape p{};
    p.crown = h.crownY;
    p.chin = h.chinY;
    const float len = h.chinY - h.crownY;
    // Corners of the gape a little below the eye line, near a real mouth's;
    // the point of the snout well below the chin, so the wedge of it clears
    // the head's own outline and can be seen.
    p.e = h.expr;
    // A smile lifts the corners of the gape without moving the point of the
    // snout, which curls the whole mouth line upward -- the same shape change
    // a person's mouth makes, read off mouthSmile and applied to a jaw that
    // has no muscles of its own.
    // A smile lifts the corners of the gape and a sad face drops them; the
    // point of the snout does not move either way, so the whole mouth line
    // curls. Both are worth several times what they were: at the old
    // amplitudes neither was visible on a head this size.
    p.cyBack = h.crownY +
               (0.56f - 0.30f * smile + 0.20f * sad) * len;
    // A smile also pulls the corners back, widening the gape.
    const float wider = 0.07f * smile;
    p.cyTip = h.chinY + 0.34f * len;
    p.botBack = h.chinY + 0.12f * len;
    p.rxMax = h.headHalfW * 1.40f;
    // Clearance over the crown, so the top of a real head stays inside the
    // shell rather than poking through the back of it.
    p.topBack = h.crownY - 0.17f * len;
    p.sHinge = 0.19f - wider;
    p.zBack = 1.20f;
    p.zTip = h.noseZ - 1.15f;
    p.girthPow = 1.75f; p.girthRoot = 0.60f; p.girthBulge = 0.08f;
    p.topPow = 1.75f; p.cyPow = 1.45f; p.botPow = 1.15f;
    p.sectionFn = coneSection;
    p.back = Vec3f(112, 108, 104);   // BGR: slate grey
    p.flank = Vec3f(150, 148, 146);
    p.belly = Vec3f(228, 231, 234);
    p.waterA = 0.36f; p.waterB = 0.53f;
    p.gullet = Vec3f(96, 104, 152);
    p.fur = 0.f;                     // skin, not fur
    p.gills = true;
    return p;
}

// Girth along the head: held through the braincase, which has a whole head to
// cover, then drawn away into the point of the muzzle.
float girthAt(float s, const HeadShape& p) {
    const float t = clampf(s, 0.f, 1.f);
    const float taper =
        std::pow(std::max(0.f, 1.f - std::pow(t, p.girthPow)), p.girthRoot);
    return taper * (1.f + p.girthBulge * std::sin(kPi * std::pow(t, 0.8f)));
}

// How far the jaw has parted from the skull at station s: 0 behind the hinge,
// where the two are one closed head, 1 in front of it, where the skull's
// cross-section is the upper half and the jaw's the lower. Blended over a
// short run so neither develops a crease at the hinge.
float gapeAt(float s, const HeadShape& p) {
    return clampf((s - p.sHinge) / 0.11f, 0.f, 1.f);
}

// The shark's: a cone, described by exponents.
Section coneSection(float s, const HeadShape& p) {
    Section c;
    const float t = clampf(s, 0.f, 1.f);
    c.z = p.zBack + (p.zTip - p.zBack) * t;
    // Top of the skull: holds the crown over the braincase, then dives along
    // the rostrum to meet the mouth line at the point.
    const float yTop = p.topBack + (p.cyTip - p.topBack) * std::pow(t, p.topPow);
    c.cy = p.cyBack + (p.cyTip - p.cyBack) * std::pow(t, p.cyPow);
    const float yBot = p.botBack + (p.cyTip - p.botBack) * std::pow(t, p.botPow);
    c.ryUp = std::max(0.f, c.cy - yTop);
    c.ryLo = std::max(0.f, yBot - c.cy);
    c.rx = p.rxMax * girthAt(t, p);
    c.gape = gapeAt(t, p);
    return c;
}

// Catmull-Rom through evenly spaced control points, clamped at both ends.
float splineAt(const float* q, float s) {
    const float t = clampf(s, 0.f, 1.f) * (kProfN - 1);
    int i = (int)t;
    if (i > kProfN - 2) i = kProfN - 2;
    const float f = t - (float)i;
    const float p0 = q[i > 0 ? i - 1 : 0];
    const float p1 = q[i], p2 = q[i + 1];
    const float p3 = q[i < kProfN - 2 ? i + 2 : kProfN - 1];
    return 0.5f * (2.f * p1 + (-p0 + p2) * f +
                   (2.f * p0 - 5.f * p1 + 4.f * p2 - p3) * f * f +
                   (-p0 + 3.f * p1 - 3.f * p2 + p3) * f * f * f);
}

// The squirrel's: a cranium and a muzzle, described by where its outline
// actually goes.
Section muzzleSection(float s, const HeadShape& p) {
    Section c;
    const float t = clampf(s, 0.f, 1.f);
    const float len = p.chin - p.crown;
    c.z = p.zBack + (p.zTip - p.zBack) * t;
    const float yTop = p.crown + splineAt(p.top, t) * len;
    c.cy = p.crown + splineAt(p.mouth, t) * len;
    const float yBot = p.crown + splineAt(p.bot, t) * len;
    c.ryUp = std::max(0.f, c.cy - yTop);
    c.ryLo = std::max(0.f, yBot - c.cy);
    c.rx = p.rxMax * std::max(0.f, splineAt(p.girth, t));
    c.gape = gapeAt(t, p);
    return c;
}

// The dinosaur's: the squirrel's control points, but with the end of the
// snout capped by a dome.
//
// Closing the outline by taking the girth to zero leaves the height where it
// was, so the head ends in a vertical blade -- invisible on the squirrel,
// whose nose sits over it, but on a broad blunt snout it showed as a crease
// down the middle of the face. Past `kCap` this holds the last full section
// and shrinks it on a quarter ellipse in every direction at once, which
// rounds the front off however the control points end.
Section domedSection(float s, const HeadShape& p) {
    constexpr float kCap = 0.86f;
    const float t = clampf(s, 0.f, 1.f);
    if (t <= kCap) return muzzleSection(t, p);
    Section c = muzzleSection(kCap, p);
    const float u = (t - kCap) / (1.f - kCap);
    const float f = std::sqrt(std::max(0.f, 1.f - u * u));
    c.z = p.zBack + (p.zTip - p.zBack) * t;
    c.rx *= f;
    c.ryUp *= f;
    c.ryLo *= f;
    c.gape = gapeAt(t, p);
    return c;
}

Section sectionAt(float s, const HeadShape& p) { return p.sectionFn(s, p); }

// Countershading: dark along the back, abruptly white underneath, which is
// the one marking that makes a grey shape read as a shark.
// Where a point sits down the model: 0 at the top of the skull, 1 at the
// point of the snout, which is its lowest part.
float depthAt(float y, const HeadShape& p) {
    // The bottom of the model, which for a cone is its point and for a
    // muzzled head is the underside of the jaw.
    const float low = std::max(p.cyTip, p.botBack);
    return clampf((y - p.topBack) / std::max(1e-3f, low - p.topBack), 0.f, 1.f);
}

// Countershading, keyed to height down the model rather than to position
// round the cross-section.
//
// Round the section is the anatomically honest choice and it does not work
// here: head-on, almost the entire visible surface is the dorsal third -- the
// belly faces the floor and shows as a hairline at the silhouette, so the
// white never appears at all. Keyed to height, the gradient runs dark at the
// top of the picture to white at the bottom, which is what reads as a shark
// from the front. The cost is that the top of the rostrum, being low, is pale
// where a real shark's is grey; from the camera's position it is not visible
// as an error.
Vec3f hideAt(float v, float s, float ax, const HeadShape& p) {
    const float t = clampf((v - p.waterA) * 7.0f, 0.f, 1.f);
    Vec3f c = p.back * (1.f - t) + p.flank * t;
    const float b = clampf((v - p.waterB) * 7.4f, 0.f, 1.f);
    c = c * (1.f - b) + p.belly * b;
    // Gill slits: five short dark bars on the flanks, behind the mouth corner
    // and never on the belly, where a shark has none.
    if (p.gills && s > 0.05f && s < 0.24f && ax > 0.26f && v > 0.26f && v < 0.54f) {
        const float ph = (s - 0.045f) / 0.032f;       // one bar per 0.032 of s
        const float d = std::fabs(ph - std::floor(ph) - 0.5f) * 2.f;
        const float slit = clampf((0.42f - d) * 5.f, 0.f, 1.f) *
                           clampf((ax - 0.30f) * 3.f, 0.f, 1.f);
        c = c * (1.f - slit * 0.85f) + Vec3f(52, 50, 54) * (slit * 0.85f);
    }
    // Bands across the back, running down the flanks and fading out before
    // the belly. Wavy rather than straight, so they read as markings on a
    // hide and not as the ribs of a tube.
    if (p.stripes > 0.f && v < p.waterB) {
        const float ph = s * 6.5f + 0.30f * std::sin(ax * 4.f);
        const float band = clampf((std::sin(ph * 2.f * kPi) - 0.35f) * 4.f, 0.f, 1.f);
        const float fade = clampf((p.waterB - v) * 6.f, 0.f, 1.f);
        c = c * (1.f - p.stripes * band * fade);
    }
    return c;
}

HeadShape squirrelShape(const Head& h) {
    // The mouth line's smile and sulk, when the mesh is not there to move it
    // point by point (see MotionMap). With it, these would count it twice.
    const float smile = mapped(h) ? 0.f : h.expr.smile;
    const float sad = mapped(h) ? 0.f : h.expr.sad;
    HeadShape p{};
    p.crown = h.crownY;
    p.chin = h.chinY;
    const float len = h.chinY - h.crownY;
    // The outline, in fractions of the head's length below the crown. Read
    // down a column to see one cross-section: the cranium (left) is tall,
    // wide and round; the muzzle (right) is a small narrow form that hangs
    // out below it.
    //
    // The muzzle *has* to hang below. Orthographic, head-on, the silhouette
    // is the union of every cross-section, so anything that stays inside the
    // biggest one -- the cranium, which has a whole head to cover -- is
    // simply not visible. A muzzle that only pointed forward would not exist
    // on screen, which is how the first attempt came out shark-shaped.
    // A near-round cranium over the first half, then a short blunt muzzle:
    // small in *both* directions. Letting the top curve dive slowly while the
    // mouth line ran on ahead gave a tall narrow spike instead of a muzzle.
    static const float kTop[kProfN]   = {-0.14f, -0.23f, -0.25f, -0.23f,
                                         -0.16f,  0.04f,  0.44f,  0.78f, 0.92f};
    static const float kMouth[kProfN] = { 0.74f,  0.74f,  0.74f,  0.75f,
                                          0.79f,  0.86f,  0.95f,  1.02f, 1.02f};
    static const float kBot[kProfN]   = { 0.92f,  0.96f,  0.97f,  0.96f,
                                          0.94f,  0.97f,  1.06f,  1.16f, 1.10f};
    // Widest at the cheeks and held there, a touch in at the back of the
    // skull, then stepping hard into the muzzle around s = 0.65.
    static const float kGirth[kProfN] = { 0.72f,  0.92f,  1.00f,  1.00f,
                                          0.94f,  0.74f,  0.48f,  0.30f, 0.f};
    for (int i = 0; i < kProfN; ++i) {
        p.top[i] = kTop[i];
        p.mouth[i] = kMouth[i];
        p.bot[i] = kBot[i];
        p.girth[i] = kGirth[i];
    }
    p.sectionFn = muzzleSection;
    // A squirrel is a round braincase with a short muzzle on the front of it,
    // not a cone: the girth holds most of the way back and then draws out
    // late. The mouth sits low and small, and the muzzle reaches only just
    // past the chin -- far enough to be seen against the head's own outline,
    // which is all the orthographic projection allows, and no further.
    p.e = h.expr;
    // A smile lifts the corners of the gape and a sad face drops them; the
    // muzzle's point stays where it is, so the line between them curls.
    // Weighted toward the corners rather than falling off linearly, which is
    // how a mouth actually changes shape.
    // Moderate on purpose. On a head described by control points the mouth
    // line also sets each cross-section's upper and lower radii, so shifting
    // it far does not curl the mouth -- it reshapes the whole head. The rest
    // of the expression goes to the ears, the eyes and the cheeks.
    const float lift = (-0.15f * smile + 0.10f * sad) * len;
    for (int i = 0; i < kProfN; ++i) {
        const float t = (float)i / (kProfN - 1);
        p.mouth[i] += lift * std::pow(1.f - t, 0.55f);
    }
    // Kept for the parts that still read them: the extremes of the outline.
    // The top of the skull is the *highest* control point, not the first one
    // -- the outline rises from the back of the head before it falls away.
    float hi = p.top[0];
    for (int i = 1; i < kProfN; ++i) hi = std::min(hi, p.top[i]);
    p.topBack = h.crownY + hi * len;
    p.cyBack = h.crownY + p.mouth[0] * len;
    p.cyTip = h.crownY + p.mouth[kProfN - 1] * len;
    p.botBack = h.crownY + p.bot[0] * len;
    p.rxMax = h.headHalfW * 1.30f;
    // A smile pulls the corners back, widening the gape.
    // Further back than the muzzle alone: with the hinge right at the
    // muzzle's root the mouth was too short for opening it to read.
    p.sHinge = 0.52f - 0.09f * smile;
    p.zBack = 1.25f;
    p.zTip = h.noseZ - 0.35f;
    p.back = Vec3f(52, 92, 148);     // BGR: chestnut along the back
    p.flank = Vec3f(88, 134, 188);
    p.belly = Vec3f(206, 226, 238);  // cream muzzle, throat and chest
    p.waterA = 0.46f; p.waterB = 0.66f;
    p.gullet = Vec3f(104, 96, 170);
    p.fur = 0.07f;                   // fur, so break the surface up a little
    p.gills = false;
    return p;
}

HeadShape elephantShape(const Head& h) {
    // The mouth line's smile and sulk, when the mesh is not there to move it
    // point by point (see MotionMap). With it, these would count it twice.
    const float smile = mapped(h) ? 0.f : h.expr.smile;
    const float sad = mapped(h) ? 0.f : h.expr.sad;
    HeadShape p{};
    p.crown = h.crownY;
    p.chin = h.chinY;
    const float len = h.chinY - h.crownY;
    // A tall domed skull, wide across the brow, coming forward and down into
    // a short face. The trunk and the ears are not in here -- they are trim,
    // and they are what actually says elephant -- so the outline only has to
    // be a big convincing head for them to hang off.
    static const float kTop[kProfN]   = {-0.16f, -0.29f, -0.32f, -0.30f,
                                         -0.22f, -0.07f,  0.13f,  0.36f, 0.50f};
    static const float kMouth[kProfN] = { 0.74f,  0.76f,  0.78f,  0.81f,
                                          0.85f,  0.90f,  0.96f,  1.01f, 1.04f};
    static const float kBot[kProfN]   = { 0.92f,  0.95f,  0.96f,  0.95f,
                                          0.93f,  0.92f,  0.95f,  1.00f, 1.06f};
    static const float kGirth[kProfN] = { 0.82f,  0.97f,  1.00f,  1.00f,
                                          0.96f,  0.88f,  0.74f,  0.52f, 0.f};
    for (int i = 0; i < kProfN; ++i) {
        p.top[i] = kTop[i];
        p.mouth[i] = kMouth[i];
        p.bot[i] = kBot[i];
        p.girth[i] = kGirth[i];
    }
    p.sectionFn = muzzleSection;
    p.e = h.expr;
    const float lift = (-0.13f * smile + 0.09f * sad) * len;
    for (int i = 0; i < kProfN; ++i) {
        const float t = (float)i / (kProfN - 1);
        p.mouth[i] += lift * std::pow(1.f - t, 0.55f);
    }
    float hi = p.top[0];
    for (int i = 1; i < kProfN; ++i) hi = std::min(hi, p.top[i]);
    p.topBack = h.crownY + hi * len;
    p.cyBack = h.crownY + p.mouth[0] * len;
    p.cyTip = h.crownY + p.mouth[kProfN - 1] * len;
    p.botBack = h.crownY + p.bot[0] * len;
    // Broad rather than tall: an elephant's head is wider than it is deep
    // and wider than it is high, and the ears hang off the width of it.
    p.rxMax = h.headHalfW * 1.62f;
    p.sHinge = 0.55f - 0.08f * smile;
    p.zBack = 1.30f;
    p.zTip = h.noseZ - 0.45f;
    p.back = Vec3f(112, 112, 118);   // BGR: elephant grey, barely warm
    p.flank = Vec3f(136, 137, 142);
    p.belly = Vec3f(164, 165, 169);  // only lightly countershaded
    p.waterA = 0.42f; p.waterB = 0.70f;
    p.gullet = Vec3f(108, 100, 168);
    p.fur = 0.05f;                   // hide, not fur, but far from smooth
    p.gills = false;
    return p;
}

HeadShape dinosaurShape(const Head& h) {
    // The mouth line's smile and sulk, when the mesh is not there to move it
    // point by point (see MotionMap). With it, these would count it twice.
    const float smile = mapped(h) ? 0.f : h.expr.smile;
    const float sad = mapped(h) ? 0.f : h.expr.sad;
    HeadShape p{};
    p.crown = h.crownY;
    p.chin = h.chinY;
    const float len = h.chinY - h.crownY;
    // A T. rex, cartoon proportions: a deep skull over a long, blunt, boxy
    // snout. As with the squirrel the snout has to *hang* -- head-on and
    // orthographic, a snout that only points forward hides inside the skull
    // -- so the mouth line and the jaw run down well past the chin, which is
    // what puts that great slab of a face on screen.
    // The front of the snout is rounded off by domedSection, past the last
    // full cross-section, so the final column only steers the spline.
    static const float kTop[kProfN]   = {-0.12f, -0.22f, -0.24f, -0.18f,
                                         -0.04f,  0.14f,  0.30f,  0.44f, 0.54f};
    static const float kMouth[kProfN] = { 0.70f,  0.72f,  0.76f,  0.82f,
                                          0.90f,  0.98f,  1.05f,  1.10f, 1.10f};
    static const float kBot[kProfN]   = { 0.94f,  1.00f,  1.04f,  1.08f,
                                          1.13f,  1.18f,  1.22f,  1.24f, 1.24f};
    // Wide at the back, where the jaw muscles are, then stepping in to a
    // snout about half as wide that holds its width and is closed off
    // bluntly -- a T. rex, rather than the shark's taper to a point. The step
    // is what makes the snout read head-on as a separate form below the
    // cheeks instead of the whole head being one egg.
    static const float kGirth[kProfN] = { 0.84f,  1.00f,  1.00f,  0.86f,
                                          0.68f,  0.58f,  0.54f,  0.50f, 0.46f};
    for (int i = 0; i < kProfN; ++i) {
        p.top[i] = kTop[i];
        p.mouth[i] = kMouth[i];
        p.bot[i] = kBot[i];
        p.girth[i] = kGirth[i];
    }
    p.sectionFn = domedSection;
    p.e = h.expr;
    // Corners up for a smile and down for a sad face; the front of the snout
    // stays put, so the line between them curls into a grin or a sulk.
    const float lift = (-0.16f * smile + 0.11f * sad) * len;
    for (int i = 0; i < kProfN; ++i) {
        const float t = (float)i / (kProfN - 1);
        p.mouth[i] += lift * std::pow(1.f - t, 0.55f);
    }
    float hi = p.top[0];
    for (int i = 1; i < kProfN; ++i) hi = std::min(hi, p.top[i]);
    p.topBack = h.crownY + hi * len;
    p.cyBack = h.crownY + p.mouth[0] * len;
    p.cyTip = h.crownY + p.mouth[kProfN - 1] * len;
    p.botBack = h.crownY + p.bot[0] * len;
    p.rxMax = h.headHalfW * 1.42f;
    // Hinged far back: a T. rex's gape runs most of the length of its head,
    // and that long row of teeth is what it is.
    p.sHinge = 0.26f - 0.06f * smile;
    p.zBack = 1.25f;
    p.zTip = h.noseZ - 1.05f;
    p.back = Vec3f(46, 104, 62);     // BGR: deep forest green
    p.flank = Vec3f(64, 150, 92);
    p.belly = Vec3f(138, 206, 196);  // pale yellow-green throat
    p.waterA = 0.40f; p.waterB = 0.66f;
    p.gullet = Vec3f(100, 96, 170);
    // Kept low: the variation is per loft vertex, and head-on the loft's
    // lines all radiate from the snout, so much more reads as rays.
    p.fur = 0.04f;
    p.gills = false;
    p.stripes = 0.34f;
    return p;
}

HeadShape dragonShape(const Head& h) {
    HeadShape p{};
    p.crown = h.crownY;
    p.chin = h.chinY;
    const float len = h.chinY - h.crownY;
    const float smile = mapped(h) ? 0.f : h.expr.smile;
    const float sad = mapped(h) ? 0.f : h.expr.sad;
    // The dinosaur's build -- a deep skull over a snout that hangs below the
    // chin, so it is seen head-on -- but leaner: the snout slopes away from
    // the brow and narrows to a muzzle, where a T. rex's stays a blunt box.
    // The front is rounded off by domedSection, as the dinosaur's is.
    static const float kTop[kProfN]   = {-0.14f, -0.25f, -0.27f, -0.20f,
                                         -0.04f,  0.16f,  0.34f,  0.50f, 0.60f};
    static const float kMouth[kProfN] = { 0.70f,  0.72f,  0.76f,  0.83f,
                                          0.92f,  1.01f,  1.09f,  1.15f, 1.16f};
    static const float kBot[kProfN]   = { 0.92f,  0.98f,  1.02f,  1.07f,
                                          1.13f,  1.19f,  1.24f,  1.27f, 1.27f};
    static const float kGirth[kProfN] = { 0.86f,  1.00f,  0.98f,  0.80f,
                                          0.60f,  0.50f,  0.45f,  0.42f, 0.38f};
    for (int i = 0; i < kProfN; ++i) {
        p.top[i] = kTop[i];
        p.mouth[i] = kMouth[i];
        p.bot[i] = kBot[i];
        p.girth[i] = kGirth[i];
    }
    p.sectionFn = domedSection;
    p.e = h.expr;
    const float lift = (-0.16f * smile + 0.11f * sad) * len;
    for (int i = 0; i < kProfN; ++i) {
        const float t = (float)i / (kProfN - 1);
        p.mouth[i] += lift * std::pow(1.f - t, 0.55f);
    }
    float hi = p.top[0];
    for (int i = 1; i < kProfN; ++i) hi = std::min(hi, p.top[i]);
    p.topBack = h.crownY + hi * len;
    p.cyBack = h.crownY + p.mouth[0] * len;
    p.cyTip = h.crownY + p.mouth[kProfN - 1] * len;
    p.botBack = h.crownY + p.bot[0] * len;
    p.rxMax = h.headHalfW * 1.36f;
    // A long gape, so the fire has a wide mouth to come out of.
    p.sHinge = 0.28f - 0.06f * smile;
    p.zBack = 1.25f;
    p.zTip = h.noseZ - 1.25f;
    p.back = Vec3f(34, 28, 140);     // BGR: deep crimson scales
    p.flank = Vec3f(46, 52, 196);
    p.belly = Vec3f(96, 196, 236);   // gold throat and jaw
    p.waterA = 0.40f; p.waterB = 0.64f;
    p.gullet = Vec3f(60, 90, 200);
    p.fur = 0.04f;
    p.gills = false;
    // Banded, like the dinosaur's hide, but finer: scales rather than stripes.
    p.stripes = 0.22f;
    return p;
}

// One half of the head: the skull if `upper`, the lower jaw otherwise.
//
// Each cross-section is a closed outline -- the outer arc, then a return along
// the mouth line -- swept along the head's axis as a single uniform grid, so
// one triangle pattern winds the whole thing and the interior of the mouth
// closes itself. Behind the hinge the return bulges out into the other half of
// the ellipse instead of running flat, which is what makes the head solid
// there rather than two shells with a slot between them.
Mesh buildHeadHalf(const HeadShape& p, bool upper) {
    Mesh m;
    m.doubleSided = true; // the open mouth shows both sides of the palate
    // Lit softly and barely glossy. At the ambient the smaller parts use, a
    // body this size is mostly surface pointing away from the light, and the
    // countershading disappears under the shading no matter what colours it
    // is given -- which is what made the white jaw look as grey as the back.
    m.ambient = 0.62f;
    m.spec = 0.16f;
    m.shin = 14;

    const Vec3f mouthCol(96, 104, 152); // BGR: the raw pink inside the mouth
    const int nArc = 19, nRet = 13, nS = 30;
    const float dir = upper ? -1.f : 1.f; // which way the outer arc bulges

    std::vector<std::vector<int>> ring(nS);
    for (int si = 0; si < nS; ++si) {
        // Bunched toward the snout, where the curvature is highest.
        const float s = std::pow((float)si / (nS - 1), 0.85f);
        const Section c = sectionAt(s, p);
        const float rOut = upper ? c.ryUp : c.ryLo;
        // The return: the far half of the ellipse behind the hinge, flattening
        // into the roof or floor of the mouth in front of it.
        const float rRet = (upper ? c.ryLo : c.ryUp) * (1.f - c.gape);

        for (int k = 0; k < nArc; ++k) {
            const float th = kPi * (float)k / (nArc - 1); // +x round to -x
            const float ca = std::cos(th), sa = std::sin(th);
            const float yw = c.cy + dir * rOut * sa;
            // Fur is not a smooth surface; a little per-vertex variation is
            // enough to stop a chestnut head reading as moulded plastic.
            const float mot = 1.f + p.fur * mottle(si, k * 7 + (int)upper);
            ring[si].push_back(m.add(Vec3f(c.rx * ca, yw, c.z),
                                     hideAt(depthAt(yw, p), s,
                                            std::fabs(ca), p) * mot));
        }
        for (int k = 1; k < nRet; ++k) {
            // Back along the other side, from -x to +x.
            const float th = kPi * (1.f - (float)k / nRet);
            const float ca = std::cos(th), sa = std::sin(th);
            // In front of the hinge this is the flat roof (or floor) of the
            // mouth, so it is coloured as flesh rather than as skin.
            const float flesh = c.gape;
            // Darker toward the back of the mouth, so the palate reads as a
            // throat going somewhere rather than as a flat red card.
            const Vec3f gullet = mouthCol * clampf(0.34f + 1.05f * s, 0.f, 1.f);
            const float yw = c.cy - dir * rRet * sa;
            const Vec3f skin = hideAt(depthAt(yw, p), s, std::fabs(ca), p);
            ring[si].push_back(m.add(Vec3f(c.rx * ca, yw, c.z),
                                     skin * (1.f - flesh) + gullet * flesh));
        }
    }

    const int nRing = nArc + nRet - 1;
    for (int si = 0; si + 1 < nS; ++si)
        for (int k = 0; k < nRing; ++k) {
            const int k2 = (k + 1) % nRing;
            m.face(ring[si][k], ring[si][k2], ring[si + 1][k2]);
            m.face(ring[si][k], ring[si + 1][k2], ring[si + 1][k]);
        }
    // Close the back of the head. The snout end needs no cap: the girth has
    // already gone to zero there.
    {
        Vec3f mid(0.f, 0.f, 0.f);
        for (int k = 0; k < nRing; ++k) mid += m.pos[ring[0][k]];
        mid = mid * (1.f / (float)nRing);
        const int c0 = m.add(mid, Vec3f(88, 86, 86));
        for (int k = 0; k < nRing; ++k)
            m.face(c0, ring[0][(k + 1) % nRing], ring[0][k]);
    }
    m.computeNormals();
    m.sortNearFirst(); // the only mesh here that occludes much of itself
    return m;
}

// A row of teeth along one jaw's mouth line: little three-sided spikes
// standing on the outer edge of the gape, pointing into the mouth.
//
// The dinosaur has them too, fewer and bigger and a shade yellower, so the
// count, the size and the colours are the caller's.
Mesh buildSharkTeeth(const HeadShape& p, bool upper, int nTooth = 11,
                     float size = 1.f, Vec3f enamel = Vec3f(238, 243, 246),
                     Vec3f root = Vec3f(196, 206, 214)) {
    Mesh m;
    m.doubleSided = true; // far too small to be worth getting winding right
    m.ambient = 0.62f;
    m.spec = 0.30f;
    m.shin = 26;
    const float dir = upper ? 1.f : -1.f; // teeth point across the gape

    for (int i = 0; i < nTooth; ++i) {
        // Spread from just inside the mouth corner to near the snout's point.
        const float s = p.sHinge + 0.10f +
                        (0.94f - p.sHinge - 0.10f) * (float)i / (nTooth - 1);
        const Section c = sectionAt(s, p);
        const float step = 0.030f;
        const Section cA = sectionAt(std::max(0.f, s - step), p);
        const Section cB = sectionAt(std::min(1.f, s + step), p);
        // Teeth shrink toward the point of the snout, as they do on a real jaw.
        const float len = size * (upper ? 0.26f : 0.22f) * (p.chin - p.cyBack) *
                          (0.55f + 0.45f * (1.f - s));
        const float half = 0.40f;

        for (float side : {-1.f, 1.f}) {
            const Vec3f a(side * (cA.rx * (1.f - half) + c.rx * half), cA.cy, cA.z);
            const Vec3f b(side * (cB.rx * (1.f - half) + c.rx * half), cB.cy, cB.z);
            const Vec3f inner(side * c.rx * 0.62f, c.cy, c.z);
            const Vec3f tipP(side * c.rx * 0.80f, c.cy + dir * len, c.z);
            const int ia = m.add(a, root), ib = m.add(b, root);
            const int ii = m.add(inner, root), it = m.add(tipP, enamel);
            // Rooted in the mouth line, and moved with it as a piece.
            m.pin(ia, (a + b) * 0.5f);
            m.face(ia, ib, it);
            m.face(ib, ii, it);
            m.face(ii, ia, it);
            m.face(ia, ii, ib);
        }
    }
    m.computeNormals();
    return m;
}

// The eyes: small black beads set into the flank, and the dorsal fin, which is
// the other half of what makes the silhouette read as a shark at a glance.
// Lies along the floor of the jaw and slides forward out of the mouth. Built
// in the jaw's frame and rotated with it, so it swings down when the jaw does
// instead of hanging in the air where the mouth used to be.
//
// `maxReach` caps how far along the mouth it gets, as a fraction of the way
// from its root to the point of the snout. The dinosaur's gape runs nearly the
// length of its head, so uncapped the tongue came out longer than the head.
Mesh buildMuzzleTongue(const HeadShape& p, float maxReach = 10.f) {
    const float s0 = p.sHinge + 0.10f;
    const Section a = sectionAt(s0, p);
    // Out past the teeth only as far as the tongue is actually out, and only
    // when there is a gap for it to come through.
    // How far out of the mouth it comes.
    //
    // tongueOut drives it when the model scores it, and a wide-open mouth
    // pushes it part way out by itself -- which is both true of a real mouth
    // and the only thing that fires reliably, since MediaPipe seldom scores
    // tongueOut above its noise floor. See the note on it in filters.cpp.
    const float gape = clampf((p.e.jawOpen - 0.55f) / 0.40f, 0.f, 1.f);
    const float out = std::max(p.e.tongue, 0.55f * gape);
    const float reach = std::min(maxReach,
                                 0.45f + 1.15f * out * (0.40f + 0.60f * p.e.jawOpen));
    const Vec3f base(0.f, a.cy, a.z);
    const Vec3f far(0.f, p.cyTip, p.zTip);
    const Vec3f tip = base + (far - base) * reach;
    return buildTongue(base, tip, 0.74f * a.rx, 0.11f * a.rx, 0.16f,
                       Vec3f(126, 112, 206));
}

// One eye bead, at a given point with a given outward direction, squashing to
// a slit as its lid closes.
//
// Split from the placement because placing an eye by its angle round the
// cross-section only works where that section is about as tall as it is wide.
// On a cranium two head-lengths tall the angle that puts an eye at the right
// height puts it right out on the silhouette, so the squirrel places its eyes
// by coordinate instead.
//
// With an `iris`, the open eye is that colour with a vertical slit of `col`
// down the middle -- a reptile's eye -- instead of solid `col`.
void addEyeBead(Mesh& m, const HeadShape& p, const Vec3f& centre,
                const Vec3f& nOut, float R, float side, const Vec3f& col,
                const Vec3f* iris = nullptr) {
    const Vec3f u = norm(nOut.cross(Vec3f(0.f, 1.f, 0.f)));
    const Vec3f v = norm(nOut.cross(u));
    // Blinking. `side` is -1 for the image-left eye, which is the one blinkL
    // describes.
    //
    // Closing squashes the bead vertically but leaves its stand-off from the
    // head alone: scaling that by the lid factor too -- as this did -- sank
    // the whole thing into the surface it sits on and a shut eye simply
    // vanished. A closed lid bulges outward anyway, it does not sink.
    // Half-lidding on a sad face, on top of any actual blink: eyes narrowed
    // a little is most of what makes an expression read as downcast.
    const float blink = clampf(((side < 0.f) ? p.e.blinkL : p.e.blinkR) +
                                   0.30f * p.e.sad,
                               0.f, 1.f);
    // Where the two lids have got to, in the same -1..1 units as the height
    // up the eye. They start clear of it and close toward each other, meeting
    // a little below the middle as real ones do.
    const float hiEdge = 1.06f - 1.24f * blink;
    const float loEdge = -1.06f + 0.94f * blink;
    // What a lid is coloured: the hide, not black. A shade darker than the
    // flank, the way a lid is shaded by the brow over it.
    const Vec3f lidCol = p.flank * 0.88f;
    // Finer than the shape alone needs: the lash line is a narrow band in
    // vOff, and at 14x6 it fell between samples and washed out to nothing.
    const int nSeg = 22, nRing = 10;
    const int first = (int)m.pos.size();
    std::vector<std::vector<int>> ring(nRing);
    for (int r = 0; r < nRing; ++r) {
        const float lat = 0.5f * kPi * (float)r / (nRing - 1);
        const float cr = std::cos(lat) * R, cz = std::sin(lat) * R;
        for (int i = 0; i < nSeg; ++i) {
            const float a = 2.f * kPi * i / nSeg;
            // Height up this point of the eye, -1 at the bottom to +1 at
            // the top, 0 at the middle. The pole is the middle of the eye,
            // which is why cos(lat) belongs in it.
            const float vOff = std::cos(lat) * std::sin(a);
            // Uncovered eyeball: below the upper lid and above the lower one.
            const float openArea = clampf((hiEdge - vOff) * 7.f, 0.f, 1.f) *
                                   clampf((vOff - loEdge) * 7.f, 0.f, 1.f);
            // The lash line, which follows the upper lid's edge down and is
            // all that is left of the eye once it is shut.
            const float lash =
                clampf(1.f - std::fabs(vOff - hiEdge) * 5.5f, 0.f, 1.f) * blink;
            const float dark = std::max(openArea, lash);
            Vec3f eye = col;
            if (iris) {
                // Across the eye, -1..1. The slit is narrow and runs the full
                // height, and it is the only dark part of an open eye.
                const float hOff = std::cos(lat) * std::cos(a);
                const float slit = clampf((0.17f - std::fabs(hOff)) * 14.f, 0.f, 1.f);
                eye = *iris * (1.f - slit) + col * slit;
                // A closed eye shows only its lash line, which stays dark.
                eye = eye * (1.f - lash) + col * lash;
            }
            ring[r].push_back(m.add(centre + u * (cr * std::cos(a)) +
                                        v * (cr * std::sin(a)) + nOut * cz,
                                    lidCol * (1.f - dark) + eye * dark));
        }
    }
    for (int r = 0; r + 1 < nRing; ++r)
        for (int i = 0; i < nSeg; ++i) {
            const int j = (i + 1) % nSeg;
            m.face(ring[r][i], ring[r][j], ring[r + 1][j]);
            m.face(ring[r][i], ring[r + 1][j], ring[r + 1][i]);
        }
    // Set into the head and carried by it, not stretched: its lids are its
    // own (above), and the face's lids moving through it would smear it.
    m.pin(first, centre);
}

// An eye on the surface of a lofted head, at a chosen depth and height.
//
// Those two are what you want to control -- how far forward the eye sits and
// how high up the face -- so they are the inputs, and the sideways position
// is *solved* from the cross-section there rather than guessed. Guessing it
// is what left the squirrel's far eye hanging in mid-air as soon as the head
// turned: a point that looks like it is on the surface from the front need
// not be on it at all.
// Where addEyeAt() puts an eye. Split out because the point-by-point mapping
// pairs the model's eyes with the face's, and has to know where they are.
Vec3f eyeCentreAt(const HeadShape& p, float sEye, float yEye, float side) {
    const Section c = sectionAt(sEye, p);
    // Where this height falls on the section's ellipse, above or below the
    // mouth line, as a fraction of the radius on that side.
    const float dy = c.cy - yEye;
    const float ry = std::max(1e-3f, dy >= 0.f ? c.ryUp : c.ryLo);
    const float n = clampf(dy / ry, -0.97f, 0.97f);
    const float fx = std::sqrt(std::max(0.f, 1.f - n * n));
    return Vec3f(side * c.rx * fx, yEye, c.z);
}

void addEyeAt(Mesh& m, const HeadShape& p, float sEye, float yEye, float R,
              float side, float faceFwd, const Vec3f& col,
              const Vec3f* iris = nullptr) {
    const Section c = sectionAt(sEye, p);
    const float dy = c.cy - yEye;
    const float ry = std::max(1e-3f, dy >= 0.f ? c.ryUp : c.ryLo);
    const float n = clampf(dy / ry, -0.97f, 0.97f);
    const float fx = std::sqrt(std::max(0.f, 1.f - n * n));
    const Vec3f centre = eyeCentreAt(p, sEye, yEye, side);
    // The ellipse's own outward normal, leaned toward the camera by
    // `faceFwd`: an eye on a rounded head should look a little forward rather
    // than straight out to the side, and the bead is anchored on the surface
    // either way.
    const Vec3f nOut = norm(Vec3f(side * fx / std::max(1e-3f, c.rx), -n / ry,
                                  -faceFwd));
    addEyeBead(m, p, centre, nOut, R, side, col, iris);
}

// A pair of eye beads set into the flank of a lofted head. `sEye` is how far
// along the head they sit and `thEye` how far round the cross-section, from
// the mouth line toward the top -- which suits a head whose sections are not
// far off round, as the shark's are.
Mesh buildEyeBeads(const HeadShape& p, float sEye, float thEye, float R,
                   const Vec3f& col, float ambient, float spec, int shin) {
    Mesh m;
    m.doubleSided = true;
    m.ambient = ambient;
    m.spec = spec;
    m.shin = shin;

    const Section ce = sectionAt(sEye, p);
    for (float side : {-1.f, 1.f}) {
        const Vec3f centre(side * ce.rx * std::cos(thEye) * 0.99f,
                           ce.cy - ce.ryUp * std::sin(thEye) * 0.99f, ce.z);
        // Outward normal of the ellipse there, so the bead sits proud of the
        // surface however the head is proportioned.
        const Vec3f nOut = norm(Vec3f(side * std::cos(thEye) / ce.rx,
                                      -std::sin(thEye) / ce.ryUp, -0.25f));
        addEyeBead(m, p, centre, nOut, R, side, col);
    }
    m.computeNormals();
    return m;
}

Mesh buildSharkTrim(const HeadShape& p) {
    Mesh m;
    m.doubleSided = true;
    m.ambient = 0.30f;
    m.spec = 0.55f;
    m.shin = 40;

    // Dorsal fin. Anatomically it belongs further down the animal, but it is
    // half of what makes a grey shape read as a shark at a glance, so it
    // stands on the back of the skull. A swept triangle with real chord and
    // thickness -- built as an outline given a left and a right copy, which
    // is why it needs no winding care.
    {
        const Section a = sectionAt(0.06f, p);
        const Section b = sectionAt(0.40f, p);
        const float rise = 0.60f * (p.cyBack - p.topBack);
        const Vec3f outline[3] = {
            Vec3f(0.f, b.cy - b.ryUp * 0.97f, b.z),                 // leading
            Vec3f(0.f, a.cy - a.ryUp * 0.97f, a.z),                 // trailing
            Vec3f(0.f, b.cy - b.ryUp * 0.97f - rise,                // tip, swept
                  b.z + 0.78f * (a.z - b.z)),
        };
        const float halfT = 0.115f * p.rxMax;
        const Vec3f finCol(118, 114, 110), tipCol(78, 76, 76);
        int L[3], R[3];
        for (int k = 0; k < 3; ++k) {
            // The fin thins to nothing at its tip and along its leading edge.
            const float th = halfT * (k == 2 ? 0.18f : 1.f);
            const Vec3f col = (k == 2) ? tipCol : finCol;
            L[k] = m.add(outline[k] + Vec3f(-th, 0.f, 0.f), col);
            R[k] = m.add(outline[k] + Vec3f(th, 0.f, 0.f), col);
        }
        m.face(L[0], L[1], L[2]);
        m.face(R[0], R[2], R[1]);
        for (int k = 0; k < 3; ++k) {
            const int k2 = (k + 1) % 3;
            m.face(L[k], R[k], R[k2]);
            m.face(L[k], R[k2], L[k2]);
        }
    }
    m.computeNormals();
    return m;
}

// --- Point-by-point mapping of the face mesh --------------------------------
//
// The blendshape scores name a dozen things a face does. The mesh shows all of
// them: every one of its 468 points has moved from where it sits at rest by
// exactly what the face is doing there, lopsided smiles, sneers, a bitten lip
// and all. So the models are moved by the mesh itself, vertex by vertex.
//
// Three steps. The face's displacements become a smooth field over the face
// (faceMotion). Each model vertex is matched to a place on the face, by a warp
// fitted between features the two have in common -- mouth corners to mouth
// corners, eyes to eyes, chin to chin -- because a shark's mouth is not where
// yours is, and moving it by whatever part of your face it happens to sit in
// front of would be meaningless. Then the displacement there is carried back
// through the inverse warp, so it arrives at the model's own scale: a smile
// that lifts your mouth corners a few millimetres lifts the corners of a gape
// three times the size of your mouth by three times as much (motionMap).
//
// The face mesh is seen from the front, so the field and the warp are both
// over the head's (x, y): the face is close to a height field that way, and
// it is the view the models are made to be seen from.

constexpr int kMeshN = 468;

// Which lip a landmark belongs to cannot be told from where it is -- with the
// mouth shut the two inner lip lines lie on top of each other -- so the lips
// are named outright, from MediaPipe's own lip contours.
constexpr int kUpperLip[] = {185, 40, 39, 37, 0, 267, 269, 270, 409,
                             191, 80, 81, 82, 13, 312, 311, 310, 415};
constexpr int kLowerLip[] = {146, 91, 181, 84, 17, 314, 405, 321, 375,
                             95, 88, 178, 87, 14, 317, 402, 318, 324};
constexpr int kLipCorner[] = {61, 291, 78, 308};
// The inner lip line, as upper and lower points matched across the mouth.
constexpr int kInnerLip[][2] = {{78, 78},   {191, 95},  {80, 88},  {81, 178},
                                {82, 87},   {13, 14},   {312, 317}, {311, 402},
                                {310, 318}, {415, 324}, {308, 308}};
// Landmarks the warp pairs with model features.
constexpr int kLmMouthR = 61, kLmMouthL = 291, kLmForehead = 10, kLmChin = 152;
constexpr int kLmNose = 1, kLmBrowR = 105, kLmBrowL = 334;
constexpr int kLmCheekR = 234, kLmCheekL = 454, kLmJawR = 172, kLmJawL = 397;
constexpr int kEyeR[] = {33, 133, 159, 145}, kEyeL[] = {263, 362, 386, 374};

// The hinge of a human jaw in the head frame: in front of the ear, a little
// below the eye line and well behind the face.
const Vec3f kJawHinge(0.f, 0.40f, 1.05f);

float smooth01(float a, float b, float x) {
    const float t = clampf((x - a) / (b - a), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

bool mapped(const Head& h) {
    return (int)h.live.size() >= kMeshN && (int)h.rest.size() >= kMeshN;
}

// A regular grid of 3D values over a rectangle of the head's (x, y), with a
// weight per node, sampled bilinearly. Zero outside.
struct Field2 {
    float x0 = 0.f, y0 = 0.f, cell = 1.f;
    int nx = 0, ny = 0;
    std::vector<Vec3f> v;
    std::vector<float> w;

    void init(float xa, float ya, float xb, float yb, float c) {
        x0 = xa; y0 = ya; cell = c;
        nx = (int)std::ceil((xb - xa) / c) + 1;
        ny = (int)std::ceil((yb - ya) / c) + 1;
        v.assign((size_t)nx * ny, Vec3f(0.f, 0.f, 0.f));
        w.assign((size_t)nx * ny, 0.f);
    }
    Vec3f sample(float x, float y, float* wOut = nullptr) const {
        if (wOut) *wOut = 0.f;
        if (nx < 2 || ny < 2) return Vec3f(0.f, 0.f, 0.f);
        const float fx = (x - x0) / cell, fy = (y - y0) / cell;
        if (!(fx >= 0.f && fy >= 0.f && fx <= nx - 1 && fy <= ny - 1))
            return Vec3f(0.f, 0.f, 0.f);
        const int i = std::min((int)fx, nx - 2), j = std::min((int)fy, ny - 2);
        const float a = fx - i, b = fy - j;
        const size_t k = (size_t)j * nx + i;
        const float c00 = (1 - a) * (1 - b), c10 = a * (1 - b);
        const float c01 = (1 - a) * b, c11 = a * b;
        if (wOut)
            *wOut = w[k] * c00 + w[k + 1] * c10 + w[k + nx] * c01 +
                    w[k + nx + 1] * c11;
        return v[k] * c00 + v[k + 1] * c10 + v[k + nx] * c01 + v[k + nx + 1] * c11;
    }
};

// The expression as two displacement fields over the face: one for what moves
// with the skull, one for what moves with the jaw.
//
// Two, because the lips cannot share one. With the mouth shut the upper and
// lower lip are a hair apart and move in opposite directions, and a single
// smooth field averages them into nothing at exactly the line where the model
// most needs to move: its mouth.
struct FaceMotion {
    Field2 upper, lower;
    bool ok = false;
};

// Gaussian splat radius for the field, in eye separations. Wide enough that
// the field has no holes between landmarks, narrow enough to keep a lip
// apart from the cheek beside it.
constexpr float kFieldSigma = 0.12f;
constexpr float kFieldCell = 0.06f;
// Splat weight at which a node counts as fully inside the face. A node in
// the middle of the face collects ~10 landmarks' worth; the field fades out
// over the last few tenths past the face's outline, rather than stopping.
constexpr float kFieldFull = 4.0f;
// Splat radius in cells: three sigmas.
constexpr int kSplatRad = 6;
static_assert(kSplatRad * kFieldCell >= 2.99f * kFieldSigma, "splat cut short");

FaceMotion faceMotion(const Head& h, bool hinged) {
    FaceMotion fm;
    if (!mapped(h)) return fm;
    const std::vector<Vec3f>& L = h.live;
    const std::vector<Vec3f>& N = h.rest;

    // How far the jaw has swung, read off the chin about the hinge, and taken
    // back out of every point that moves with the jaw. The model has a hinge
    // of its own, opened by jawOpen, and a model jaw that swung *and* was
    // pushed down by the same motion would open twice. What is left is
    // everything the jaw does besides swinging: thrust, a sideways shift, the
    // lower lip curling or dropping on its own. A model without a hinge --
    // the painted animals' parts -- keeps the swing, since nothing else will
    // open its mouth.
    auto angYZ = [](const Vec3f& d) { return std::atan2(d[2], d[1]); };
    const float theta = hinged ? angYZ(L[kLmChin] - kJawHinge) -
                                     angYZ(N[kLmChin] - kJawHinge)
                               : 0.f;

    // How much each landmark belongs to the jaw rather than the skull, from
    // where it sits on the face at rest. Below the line between the lips it
    // is the jaw's; above, the skull's. Out past the mouth corners there is
    // no lip line to go by, so the cheeks change hands gradually, which is
    // what they do on a real face.
    float lipX[11], lipY[11];
    for (int k = 0; k < 11; ++k) {
        const Vec3f m = (N[kInnerLip[k][0]] + N[kInnerLip[k][1]]) * 0.5f;
        lipX[k] = m[0];
        lipY[k] = m[1];
    }
    if (lipX[0] > lipX[10]) { // the line runs whichever way the image does
        std::reverse(lipX, lipX + 11);
        std::reverse(lipY, lipY + 11);
    }
    auto lipLineY = [&](float x) {
        if (x <= lipX[0]) return lipY[0];
        for (int k = 0; k < 10; ++k)
            if (x <= lipX[k + 1]) {
                const float t = (x - lipX[k]) / std::max(1e-4f, lipX[k + 1] - lipX[k]);
                return lipY[k] + (lipY[k + 1] - lipY[k]) * t;
            }
        return lipY[10];
    };
    const float mouthY = lipY[5];
    const float cornerX = 0.5f * (lipX[10] - lipX[0]);

    float jawW[kMeshN];
    for (int i = 0; i < kMeshN; ++i) {
        const float x = N[i][0], y = N[i][1];
        const float inMouth = smooth01(cornerX + 0.15f, cornerX - 0.05f, std::fabs(x));
        const float yl = lipLineY(x);
        jawW[i] = inMouth * smooth01(yl - 0.02f, yl + 0.05f, y) +
                  (1.f - inMouth) * smooth01(mouthY - 0.05f, mouthY + 0.35f, y);
    }
    for (int i : kUpperLip) jawW[i] = 0.f;
    for (int i : kLowerLip) jawW[i] = 1.f;
    for (int i : kLipCorner) jawW[i] = 0.5f;

    fm.upper.init(-1.9f, -1.4f, 1.9f, 2.4f, kFieldCell);
    fm.lower.init(-1.9f, -1.4f, 1.9f, 2.4f, kFieldCell);
    const int rad = kSplatRad;
    const float inv2s2 = 1.f / (2.f * kFieldSigma * kFieldSigma);
    for (int i = 0; i < kMeshN; ++i) {
        // Where the point would be if the jaw had only swung.
        const float a = jawW[i] * theta;
        const float ca = std::cos(a), sa = std::sin(a);
        const Vec3f d = N[i] - kJawHinge;
        const Vec3f swung = kJawHinge + Vec3f(d[0], ca * d[1] - sa * d[2],
                                              sa * d[1] + ca * d[2]);
        const Vec3f D = L[i] - swung;

        const float px = N[i][0], py = N[i][1];
        const int ci = (int)std::lround((px - fm.upper.x0) / kFieldCell);
        const int cj = (int)std::lround((py - fm.upper.y0) / kFieldCell);
        const int k0 = std::max(0, ci - rad), k1 = std::min(fm.upper.nx - 1, ci + rad);
        const int j0 = std::max(0, cj - rad), j1 = std::min(fm.upper.ny - 1, cj + rad);
        if (k0 > k1 || j0 > j1) continue;
        // A Gaussian is the product of one along each axis, so the kernel is
        // two short rows of exponentials rather than a square of them.
        float gx[2 * kSplatRad + 1], gy[2 * kSplatRad + 1];
        for (int k = k0; k <= k1; ++k) {
            const float dx = fm.upper.x0 + k * kFieldCell - px;
            gx[k - k0] = std::exp(-dx * dx * inv2s2);
        }
        for (int j = j0; j <= j1; ++j) {
            const float dy = fm.upper.y0 + j * kFieldCell - py;
            gy[j - j0] = std::exp(-dy * dy * inv2s2);
        }
        for (int j = j0; j <= j1; ++j)
            for (int k = k0; k <= k1; ++k) {
                const float g = gx[k - k0] * gy[j - j0];
                if (g < 0.01f) continue;
                const size_t n = (size_t)j * fm.upper.nx + k;
                const float gu = g * (1.f - jawW[i]), gl = g * jawW[i];
                fm.upper.v[n] += D * gu;
                fm.upper.w[n] += gu;
                fm.lower.v[n] += D * gl;
                fm.lower.w[n] += gl;
            }
    }
    // Normalise: the value is the weighted mean of the nearby displacements,
    // and the weight becomes how much of the face is under this node, 0..1.
    for (Field2* f : {&fm.upper, &fm.lower})
        for (size_t n = 0; n < f->v.size(); ++n) {
            const float s = f->w[n];
            f->v[n] = s > 1e-4f ? f->v[n] * (1.f / s) : Vec3f(0.f, 0.f, 0.f);
            f->w[n] = smooth01(0.f, kFieldFull, s);
        }
    fm.ok = true;
    return fm;
}

// A thin-plate spline in the plane: the smoothest warp that carries a set of
// points onto another set. With no points it is the identity.
struct Tps {
    std::vector<cv::Point2f> c;
    std::vector<float> wx, wy;
    float ax[3] = {0.f, 1.f, 0.f}, ay[3] = {0.f, 0.f, 1.f};

    static float U(float r2) { return r2 > 1e-12f ? 0.5f * r2 * std::log(r2) : 0.f; }

    cv::Point2f operator()(cv::Point2f p) const {
        float x = ax[0] + ax[1] * p.x + ax[2] * p.y;
        float y = ay[0] + ay[1] * p.x + ay[2] * p.y;
        for (size_t i = 0; i < c.size(); ++i) {
            const float dx = p.x - c[i].x, dy = p.y - c[i].y;
            const float u = U(dx * dx + dy * dy);
            x += wx[i] * u;
            y += wy[i] * u;
        }
        return cv::Point2f(x, y);
    }

    // `stiff` trades exactness at the anchors for smoothness between them, so
    // two anchors that crowd each other bend the warp rather than fold it.
    bool fit(const std::vector<cv::Point2f>& src,
             const std::vector<cv::Point2f>& dst, float stiff) {
        const int n = (int)src.size(), m = n + 3;
        if (n < 3) return false;
        std::vector<double> A((size_t)m * m, 0.0), bx(m, 0.0), by(m, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const float dx = src[i].x - src[j].x, dy = src[i].y - src[j].y;
                A[(size_t)i * m + j] = U(dx * dx + dy * dy) + (i == j ? stiff : 0.f);
            }
            const double P[3] = {1.0, src[i].x, src[i].y};
            for (int k = 0; k < 3; ++k) {
                A[(size_t)i * m + n + k] = P[k];
                A[(size_t)(n + k) * m + i] = P[k];
            }
            bx[i] = dst[i].x;
            by[i] = dst[i].y;
        }
        // Gaussian elimination with partial pivoting; m is a dozen or two.
        for (int col = 0; col < m; ++col) {
            int piv = col;
            for (int r = col + 1; r < m; ++r)
                if (std::fabs(A[(size_t)r * m + col]) > std::fabs(A[(size_t)piv * m + col]))
                    piv = r;
            if (std::fabs(A[(size_t)piv * m + col]) < 1e-12) return false;
            if (piv != col) {
                for (int k = 0; k < m; ++k)
                    std::swap(A[(size_t)col * m + k], A[(size_t)piv * m + k]);
                std::swap(bx[col], bx[piv]);
                std::swap(by[col], by[piv]);
            }
            for (int r = col + 1; r < m; ++r) {
                const double f = A[(size_t)r * m + col] / A[(size_t)col * m + col];
                if (f == 0.0) continue;
                for (int k = col; k < m; ++k) A[(size_t)r * m + k] -= f * A[(size_t)col * m + k];
                bx[r] -= f * bx[col];
                by[r] -= f * by[col];
            }
        }
        for (int r = m - 1; r >= 0; --r) {
            for (int k = r + 1; k < m; ++k) {
                bx[r] -= A[(size_t)r * m + k] * bx[k];
                by[r] -= A[(size_t)r * m + k] * by[k];
            }
            bx[r] /= A[(size_t)r * m + r];
            by[r] /= A[(size_t)r * m + r];
        }
        c = src;
        wx.assign(n, 0.f);
        wy.assign(n, 0.f);
        for (int i = 0; i < n; ++i) {
            wx[i] = (float)bx[i];
            wy[i] = (float)by[i];
        }
        for (int k = 0; k < 3; ++k) {
            ax[k] = (float)bx[n + k];
            ay[k] = (float)by[n + k];
        }
        return true;
    }
};

// Features a model shares with the face, as matched pairs in the head's
// (x, y): the model's on the left, the face's at rest on the right.
struct Anchors {
    std::vector<cv::Point2f> model, face;
    void add(cv::Point2f m, cv::Point2f f) {
        model.push_back(m);
        face.push_back(f);
    }
    // A left/right pair. The face's two are put in image order, which the
    // mesh's own subject-side labels do not settle.
    void addPair(cv::Point2f mRight, cv::Point2f fa, cv::Point2f fb) {
        if (fa.x > fb.x) std::swap(fa, fb);
        add(cv::Point2f(-mRight.x, mRight.y), fa);
        add(mRight, fb);
    }
};

// The model's displacement, at every point of its own (x, y): the face's
// motion, carried across the warp and back at the model's scale.
struct MotionMap {
    Field2 upper, lower;
    bool ok = false;
};

// How far the warp may miss its anchors to stay smooth between them. Enough
// that the elephant's, whose eyes sit nearly level with its mouth corners,
// bends instead of folding.
constexpr float kWarpStiff = 0.10f;

MotionMap motionMap(const FaceMotion& fm, const Anchors& an,
                    const std::vector<const Mesh*>& meshes, float gain) {
    MotionMap mm;
    if (!fm.ok) return mm;
    Tps toFace, toModel;
    const bool warped = !an.model.empty();
    if (warped && (!toFace.fit(an.model, an.face, kWarpStiff) ||
                   !toModel.fit(an.face, an.model, kWarpStiff)))
        return mm;
    // Depth has no anchors of its own, so it is scaled by the warp's overall
    // magnification: a model whose features sit twice as far apart as the
    // face's moves twice as far front-to-back too.
    float zScale = 1.f;
    if (warped) {
        const float det = toFace.ax[1] * toFace.ay[2] - toFace.ax[2] * toFace.ay[1];
        zScale = clampf(1.f / std::sqrt(std::max(1e-4f, std::fabs(det))), 0.5f, 4.f);
    }

    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (const Mesh* m : meshes) {
        for (const Vec3f& p : m->pos) {
            x0 = std::min(x0, p[0]); x1 = std::max(x1, p[0]);
            y0 = std::min(y0, p[1]); y1 = std::max(y1, p[1]);
        }
    }
    if (x1 <= x0 || y1 <= y0) return mm;
    // Fine enough that a lip's worth of motion keeps its shape, coarse
    // enough that the per-node warp costs little next to the raster: each
    // node is three spline evaluations, and the motion is smooth anyway.
    const float cell = std::max(0.04f, std::max(x1 - x0, y1 - y0) / 40.f);
    mm.upper.init(x0 - cell, y0 - cell, x1 + cell, y1 + cell, cell);
    mm.lower.init(x0 - cell, y0 - cell, x1 + cell, y1 + cell, cell);
    for (int j = 0; j < mm.upper.ny; ++j)
        for (int i = 0; i < mm.upper.nx; ++i) {
            const cv::Point2f p(mm.upper.x0 + i * cell, mm.upper.y0 + j * cell);
            const cv::Point2f f = warped ? toFace(p) : p;
            const cv::Point2f back = warped ? toModel(f) : f;
            const size_t n = (size_t)j * mm.upper.nx + i;
            const Field2* src[2] = {&fm.upper, &fm.lower};
            Field2* dst[2] = {&mm.upper, &mm.lower};
            for (int k = 0; k < 2; ++k) {
                float w = 0.f;
                const Vec3f D = src[k]->sample(f.x, f.y, &w);
                if (w <= 1e-3f) continue;
                const cv::Point2f to =
                    warped ? toModel(cv::Point2f(f.x + D[0], f.y + D[1]))
                           : cv::Point2f(f.x + D[0], f.y + D[1]);
                dst[k]->v[n] = Vec3f(to.x - back.x, to.y - back.y, D[2] * zScale) *
                               (w * gain);
                dst[k]->w[n] = w;
            }
        }
    mm.ok = true;
    return mm;
}

// The least-squares affine map from `src` to `dst`: out = (A0 + A1 x + A2 y,
// A3 + A4 x + A5 y). False when the points do not span the plane.
bool fitAffine(const std::vector<cv::Point2f>& src,
               const std::vector<cv::Point2f>& dst, float A[6]) {
    // Normal equations, shared by both output coordinates.
    double M[3][3] = {}, bx[3] = {}, by[3] = {};
    for (size_t i = 0; i < src.size(); ++i) {
        const double r[3] = {1.0, src[i].x, src[i].y};
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) M[j][k] += r[j] * r[k];
            bx[j] += r[j] * dst[i].x;
            by[j] += r[j] * dst[i].y;
        }
    }
    const double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                       M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                       M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    if (std::fabs(det) < 1e-9) return false;
    // Cramer's rule: three unknowns, so nothing cleverer is worth it.
    auto solve = [&](const double* b, float* out) {
        for (int c = 0; c < 3; ++c) {
            double T[3][3];
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) T[j][k] = (k == c) ? b[j] : M[j][k];
            const double d = T[0][0] * (T[1][1] * T[2][2] - T[1][2] * T[2][1]) -
                             T[0][1] * (T[1][0] * T[2][2] - T[1][2] * T[2][0]) +
                             T[0][2] * (T[1][0] * T[2][1] - T[1][1] * T[2][0]);
            out[c] = (float)(d / det);
        }
    };
    solve(bx, A);
    solve(by, A + 3);
    return true;
}

// A lofted head's features, paired with the same features on the face.
//
// The model's mouth is the loft's mouth line seen from the front: corners at
// the hinge, sweeping round to the point of the snout. Your mouth is nearly
// straight across, so the warp has to bend one into the other, and the
// points along it are what pin that bend down -- corners, a point halfway
// out along each side, and the middle. With the eyes, the top of the head
// and the chin, those are features the model really has, and they are
// matched exactly.
//
// Everything else on the face -- nose, brows, cheeks, the angle of the jaw --
// has no counterpart on a shark, and matching it to some invented point
// folds the warp: the shark's gape runs up beside its eyes, so wherever its
// "cheek" is put, it lands almost on top of the mouth corner, and the inverse
// of a warp that squeezes that much face into that little model throws the
// teeth off the side of the head. So those are carried across by the overall
// fit of model to face instead (fitAffine below). They hold the warp to that
// fit away from the features, rather than adding bends of their own.
Anchors loftAnchors(const Head& h, const HeadShape& p, const Vec3f& eye) {
    Anchors a;
    const std::vector<Vec3f>& N = h.rest;
    auto F = [&](int i) { return cv::Point2f(N[i][0], N[i][1]); };
    auto Avg = [&](const int* ids, int n) {
        cv::Point2f s(0.f, 0.f);
        for (int k = 0; k < n; ++k) s = s + F(ids[k]);
        return s * (1.f / (float)n);
    };
    const cv::Point2f fEyeA = Avg(kEyeR, 4), fEyeB = Avg(kEyeL, 4);

    const Section corner = sectionAt(p.sHinge, p);
    const Section front = sectionAt(1.f, p);
    // Halfway out along the mouth line: the station whose width is half the
    // corner's. Scanned rather than solved, since the outline is a spline.
    Section half = corner;
    float lowest = corner.cy + corner.ryLo;
    for (int k = 0; k <= 48; ++k) {
        const Section c = sectionAt((float)k / 48.f, p);
        lowest = std::max(lowest, c.cy + c.ryLo);
    }
    for (int k = 0; k <= 48; ++k) {
        const float s = p.sHinge + (1.f - p.sHinge) * (float)k / 48.f;
        const Section c = sectionAt(s, p);
        if (c.rx <= 0.5f * corner.rx) { half = c; break; }
    }

    a.addPair(cv::Point2f(corner.rx, corner.cy), F(kLmMouthR), F(kLmMouthL));
    a.addPair(cv::Point2f(half.rx, half.cy), (F(81) + F(178)) * 0.5f,
              (F(311) + F(402)) * 0.5f);
    a.add(cv::Point2f(0.f, front.cy), (F(13) + F(14)) * 0.5f);
    a.addPair(cv::Point2f(std::fabs(eye[0]), eye[1]), fEyeA, fEyeB);
    a.add(cv::Point2f(0.f, p.topBack), F(kLmForehead));
    a.add(cv::Point2f(0.f, lowest), F(kLmChin));

    // The rest of the face, carried across by the fit of the features above.
    float A[6];
    if (!fitAffine(a.face, a.model, A)) return a;
    auto carry = [&](cv::Point2f f) {
        return cv::Point2f(A[0] + A[1] * f.x + A[2] * f.y,
                           A[3] + A[4] * f.x + A[5] * f.y);
    };
    for (int i : {kLmNose, kLmBrowR, kLmBrowL, kLmCheekR, kLmCheekL, kLmJawR,
                  kLmJawL})
        a.add(carry(F(i)), F(i));
    return a;
}

// How much the lofted heads exaggerate the face's motion.
constexpr float kLoftGain = 1.5f;

// How much of the motion reaches a point at depth z. The face is at the front
// of the model; the back of a skull a head's depth behind it is not, and
// moving it with the brows would only ripple the silhouette.
constexpr float kMotionZNear = 0.80f, kMotionZFar = 1.80f;

void deform(Mesh& m, const MotionMap& mm) {
    if (!mm.ok) return;
    const Field2& f = m.jaw ? mm.lower : mm.upper;
    bool moved = false;
    for (size_t i = 0; i < m.pos.size(); ++i) {
        const Vec3f& q = i < m.bind.size() ? m.bind[i] : m.pos[i];
        const float wz = smooth01(kMotionZFar, kMotionZNear, q[2]);
        if (wz <= 0.f) continue;
        const Vec3f d = f.sample(q[0], q[1]);
        if (d.dot(d) < 1e-10f) continue;
        m.pos[i] += d * wz;
        moved = true;
    }
    if (moved) m.computeNormals();
}

// --- Fire -----------------------------------------------------------------------
// The dragon's breath: a jet of flame from between its jaws, drawn over the
// model as particles rather than as geometry. Fire has no surface to shade.
// Each particle deposits *heat*, and the summed heat is coloured through a
// fire palette: too little is nothing at all, then deep red, orange, yellow
// and a white core where the most flames crowd together. Colouring the
// particles themselves and adding them up was tried first; it averages into
// a soft ball with a pink haze round it, because there is no threshold to give
// the flames edges and faint red screened over a pale background is pink.
struct Fire {
    float amount = 0.f; // 0 none .. 1 full blast
    Vec3f origin;       // model space: between the jaws
    Vec3f dir;          // unit: the way the jet goes
    float length = 0.f; // how far it reaches, in eye separations
};

constexpr int kFlames = 170;
// Frames from a flame leaving the mouth to burning out: about two thirds of a
// second at the preview's rate. Every flame is somewhere in that cycle, so
// the jet is always full and always moving.
constexpr float kFlameLife = 20.f;

struct Flame {
    Vec3f p;    // model space
    float r;    // radius, eye separations
    float t;    // age, 0 at the mouth .. 1 burnt out
    float heat; // brightness, 0..1
};

float hash01(int i, int k) { return 0.5f * (mottle(i, k * 131 + 17) + 1.f); }

// Every flame of `f` at this frame. One generator for drawing and for bounds,
// so the region the preview prepares is exactly the one the flames land in.
template <class Fn> void forEachFlame(const Head& h, const Fire& f, Fn fn) {
    if (f.amount <= 0.f || f.length <= 0.f) return;
    // Across the jet. The jet always lies in the head's y-z plane, so +x is
    // square to it.
    const Vec3f u(1.f, 0.f, 0.f);
    const Vec3f v = norm(f.dir.cross(u));
    const float L = f.length;
    for (int i = 0; i < kFlames; ++i) {
        const float a = (float)h.phase / kFlameLife + hash01(i, 1);
        const float t = a - std::floor(a);
        // Leaves the mouth fast and slows as it spreads, the way a jet does.
        const float along = L * std::pow(t, 0.8f);
        // A cone that widens with age, each flame at its own place in it and
        // swirling as it goes so the jet churns instead of streaming.
        const float R = L * (0.06f + 0.55f * t);
        const float th = 2.f * kPi * hash01(i, 2) + 2.4f * t;
        const float rho = std::sqrt(hash01(i, 3));
        Vec3f q = f.origin + f.dir * along +
                  u * (std::cos(th) * rho * R) + v * (std::sin(th) * rho * R);
        // Hot gas rises: the end of the jet lifts.
        q[1] -= 0.20f * L * t * t;
        const float r = L * (0.045f + 0.17f * t) * (0.7f + 0.6f * hash01(i, 4));
        // Fades in over the first few frames so flames do not pop into being
        // at the lip, and cools as it burns, flickering on the way. Young
        // flames crowd the mouth and old ones spread, so the heat is highest
        // at the lip and falls away down the jet -- which is what puts the
        // white core at the mouth and the red at the edges.
        const float flicker = 0.65f + 0.35f * std::sin((float)h.phase * 0.9f + 7.f * i);
        const float heat = 0.45f * f.amount * flicker *
                           std::min(1.f, t * 12.f) * std::pow(1.f - t, 1.1f);
        fn(Flame{q, r, t, heat});
    }
}

// The fire palette: summed heat to colour (BGR), deep red through orange and
// yellow to white.
Vec3f firePalette(float H) {
    static const float stop[5] = {0.10f, 0.35f, 0.75f, 1.30f, 2.40f};
    static const Vec3f col[5] = {Vec3f(20, 25, 160), Vec3f(15, 90, 240),
                                 Vec3f(20, 160, 255), Vec3f(60, 225, 255),
                                 Vec3f(210, 250, 255)};
    if (H <= stop[0]) return col[0];
    for (int k = 0; k < 4; ++k)
        if (H <= stop[k + 1]) {
            const float w = (H - stop[k]) / (stop[k + 1] - stop[k]);
            return col[k] * (1.f - w) + col[k + 1] * w;
        }
    return col[4];
}

// How much further a flame reaches above its centre than below it, on
// screen. A flame is a teardrop that licks upward; round splats sum to a
// cauliflower, and the same splats drawn tall at the top sum to tongues.
constexpr float kFlameTall = 1.8f;

// Where a flame lands on screen, and how far its glow reaches there.
void flameOnScreen(const Head& h, const Flame& fl, float& sx, float& sy,
                   float& sr) {
    const Vec3f w = h.R * (fl.p * h.unit);
    sx = h.anchor.x + w[0];
    sy = h.anchor.y + w[1];
    sr = fl.r * h.unit;
}

cv::Rect fireBounds(const Head& h, const Fire& f) {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    forEachFlame(h, f, [&](const Flame& fl) {
        if (fl.heat <= 0.f) return;
        float sx, sy, sr;
        flameOnScreen(h, fl, sx, sy, sr);
        x0 = std::min(x0, sx - sr); x1 = std::max(x1, sx + sr);
        y0 = std::min(y0, sy - kFlameTall * sr); y1 = std::max(y1, sy + sr);
    });
    if (x1 <= x0 || y1 <= y0) return cv::Rect();
    return cv::Rect((int)std::floor(x0) - 2, (int)std::floor(y0) - 2,
                    (int)std::ceil(x1 - x0) + 4, (int)std::ceil(y1 - y0) + 4);
}

// Downsampling for the heat field. Flames are soft, so a quarter of the
// resolution loses nothing visible, and it is sixteen times less work: at
// full size the splats alone cost more than the whole dragon's raster.
constexpr int kFireDown = 4;

void drawFire(cv::Mat& frame, const Head& h, const Fire& f) {
    const cv::Rect box = fireBounds(h, f) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (box.width < 2 || box.height < 2) return;
    const int LW = box.width / kFireDown + 2, LH = box.height / kFireDown + 2;
    std::vector<float> heat((size_t)LW * LH, 0.f);
    std::vector<float> smoke((size_t)LW * LH, 0.f);

    forEachFlame(h, f, [&](const Flame& fl) {
        if (fl.heat <= 0.f) return;
        float sx, sy, sr;
        flameOnScreen(h, fl, sx, sy, sr);
        const float cx = (sx - box.x) / kFireDown, cy = (sy - box.y) / kFireDown;
        const float R = std::max(1.f, sr / kFireDown);
        // Old flames leave a little smoke behind them.
        const float sm = 0.06f * f.amount * clampf((fl.t - 0.65f) * 3.f, 0.f, 1.f);
        const int ya = std::max(0, (int)std::floor(cy - kFlameTall * R));
        const int yb = std::min(LH - 1, (int)std::ceil(cy + R));
        const int xa = std::max(0, (int)std::floor(cx - R));
        const int xb = std::min(LW - 1, (int)std::ceil(cx + R));
        const float inv = 1.f / (R * R);
        for (int y = ya; y <= yb; ++y)
            for (int x = xa; x <= xb; ++x) {
                const float dx = x - cx;
                // Above the centre the flame is drawn out into a tongue,
                // narrowing as it rises.
                const float dy = (y < cy) ? (y - cy) / kFlameTall : (y - cy);
                const float d2 = (dx * dx * (y < cy ? 1.f + 0.6f * -dy / R : 1.f) +
                                  dy * dy) * inv;
                if (d2 >= 1.f) continue;
                // A soft round falloff without an exp per pixel.
                const float k = (1.f - d2) * (1.f - d2);
                const size_t n = (size_t)y * LW + x;
                heat[n] += fl.heat * k;
                smoke[n] += sm * k;
            }
    });

    // Onto the frame. Smoke darkens what is behind it; then the flames go
    // over it in their palette colour, opaque where the heat is enough to burn
    // and fading out just below that, which is what gives them edges; and a
    // faint orange glow is screened round them, so they light the air instead
    // of being cut out of paper.
    //
    // All three steps are, per channel, linear in what is underneath --
    // screening by c is v (1 - c/255) + c -- so together they are one
    // multiply-add, `v m + b`, and m and b depend only on the heat and the
    // smoke. They are worked out once per node of the coarse grid and only
    // interpolated per pixel: the palette and its threshold per pixel, at full
    // resolution over a fireball, cost more than the splatting did.
    const Vec3f smokeCol(40, 40, 46);
    const Vec3f haloCol(10, 70, 210);
    // Per node: the three multipliers, then the three offsets.
    std::vector<float> ma((size_t)LW * LH * 6, 0.f);
    std::vector<uchar> lit((size_t)LW * LH, 0);
    for (size_t n = 0; n < heat.size(); ++n) {
        float* q = &ma[n * 6];
        q[0] = q[1] = q[2] = 1.f;
        const float H = heat[n];
        const float sm = std::min(0.55f, smoke[n]);
        if (H < 0.01f && sm < 0.004f) continue;
        const float alpha = smooth01(0.10f, 0.32f, H);
        const Vec3f pal = firePalette(H);
        const float halo = 0.45f * std::min(1.f, H * 2.5f) * (1.f - alpha);
        for (int k = 0; k < 3; ++k) {
            const float c = haloCol[k] * halo;
            const float keep = 1.f - c * (1.f / 255.f);
            q[k] = (1.f - sm) * (1.f - alpha) * keep;
            q[3 + k] = (smokeCol[k] * sm * (1.f - alpha) + pal[k] * alpha) * keep + c;
        }
        lit[n] = 1;
    }
    // Each column's place in the coarse grid, once rather than per row.
    std::vector<int> colI(box.width);
    std::vector<float> colA(box.width);
    for (int x = 0; x < box.width; ++x) {
        const float fx = std::min((float)(LW - 1), (float)x / kFireDown);
        colI[x] = std::min(LW - 2, (int)fx);
        colA[x] = fx - colI[x];
    }
    // Interpolated down the column once per output row, so each pixel only
    // blends two nodes across -- at full resolution over a fireball, the
    // per-pixel work is the whole cost of the fire.
    std::vector<float> row((size_t)LW * 6);
    std::vector<uchar> rowLit(LW);
    for (int y = 0; y < box.height; ++y) {
        const float fy = std::min((float)(LH - 1), (float)y / kFireDown);
        const int j = std::min(LH - 2, (int)fy);
        const float b = fy - j;
        const float* r0 = &ma[(size_t)j * LW * 6];
        const float* r1 = r0 + (size_t)LW * 6;
        bool any = false;
        for (int i = 0; i < LW; ++i) {
            rowLit[i] = lit[(size_t)j * LW + i] | lit[(size_t)(j + 1) * LW + i];
            any |= rowLit[i] != 0;
            for (int k = 0; k < 6; ++k)
                row[i * 6 + k] = r0[i * 6 + k] + (r1[i * 6 + k] - r0[i * 6 + k]) * b;
        }
        if (!any) continue;
        uchar* d = frame.ptr<uchar>(box.y + y) + 3 * box.x;
        for (int x = 0; x < box.width; ++x, d += 3) {
            const int i = colI[x];
            if (!(rowLit[i] | rowLit[i + 1])) continue;
            const float a = colA[x];
            const float* p0 = &row[i * 6];
            const float* p1 = p0 + 6;
            for (int k = 0; k < 3; ++k) {
                const float m = p0[k] + (p1[k] - p0[k]) * a;
                const float o = p0[3 + k] + (p1[3 + k] - p0[3 + k]) * a;
                const float v = d[k] * m + o;
                d[k] = (uchar)(v <= 0.f ? 0 : (v >= 255.f ? 255 : (int)(v + 0.5f)));
            }
        }
    }
}

// --- Projection + rasteriser -----------------------------------------------

// Orthographic: the basis already carries the head's real 3D orientation, so
// the image offset is just the rotated model offset and its z is the depth.
struct Proj { cv::Point2f s; float depth; };
Proj project(const Head& h, const Vec3f& world, float ssScale, cv::Point2f ssOrg) {
    Proj p;
    p.s = (h.anchor + cv::Point2f(world[0], world[1]) - ssOrg) * ssScale;
    p.depth = world[2];
    return p;
}

// Light rig (camera space): key light from the upper-left front.
// x to a small non-negative integer power, by squaring.
//
// The specular term evaluates this once per supersample, which for a model
// the size of the shark's head is millions of times a frame, and std::pow is
// a general transcendental that cannot know the exponent is a constant 14 or
// 26. Worth 12 ms a frame on its own, for identical output.
inline float fastPow(float x, int n) {
    float r = 1.f;
    while (n) {
        if (n & 1) r *= x;
        x *= x;
        n >>= 1;
    }
    return r;
}

const Vec3f kLight = norm(Vec3f(-0.4f, -0.55f, -0.8f));
const Vec3f kHalf = norm(kLight + Vec3f(0.f, 0.f, -1.f)); // view = -Z

// Rasterise one mesh into the supersampled layer with a shared z-buffer.
// `layer` is BGR float, `cover` marks written pixels, `zbuf` resolves occlusion.
void raster(const Mesh& m, const Head& h, float ssScale, cv::Point2f ssOrg,
            cv::Mat& layer, cv::Mat& cover, cv::Mat& zbuf) {
    const int W = layer.cols, H = layer.rows;
    std::vector<Vec3f> world(m.pos.size()), wnrm(m.nrm.size());
    std::vector<Proj> pr(m.pos.size());
    for (size_t i = 0; i < m.pos.size(); ++i) {
        world[i] = h.R * (m.pos[i] * h.unit);
        wnrm[i] = h.R * m.nrm[i]; // R is orthonormal; normals need no rescale
        pr[i] = project(h, world[i], ssScale, ssOrg);
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
                // Turn the normal to the side we are actually looking at.
                //
                // Note the convention: screen y runs down, so the sign of the
                // area test comes out inverted from the usual one and these
                // meshes wind so that a *front* face's normal points away.
                // Hence the flip is on !back.
                //
                // Keyed off the winding rather than on the normal's own sign,
                // which is what the old `if (n[2] > 0)` did: a carved surface
                // has front faces whose normals tilt away from the camera --
                // the far wall of a nostril -- and flipping those lights the
                // inside of the pit exactly like its outside, which turned
                // the nostrils into flat gashes.
                if (!back) n = -n;
                Vec3f base = w0 * m.col[t[0]] + w1 * m.col[t[1]] + w2 * m.col[t[2]];

                float diff = std::max(0.f, n.dot(kLight));
                float shade = m.ambient + (1.f - m.ambient) * diff;
                float s = m.spec * fastPow(std::max(0.f, n.dot(kHalf)), m.shin);
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

// Everything a squirrel has that the loft does not: ears on top of the
// braincase, big forward eyes, a nose on the point of the muzzle, the buck
// teeth, and cheek pouches that fill on cheekPuff.
void addSquirrelTrim(std::vector<Mesh>& out, const Head& head,
                     const HeadShape& p) {
    const Style st = styleFor(Species::Squirrel);
    const float len = head.chinY - head.crownY;

    // The ears belong on the *model's* skull, which is wider and taller than
    // the real head inside it, so they are built against a head measured from
    // the shape rather than from the face.
    Head eh = head;
    // Set so the lobes' bases sit just *inside* the cranium's dome and the
    // ears grow out of the top of it. Wider or higher and they perch above
    // the head with daylight under them.
    eh.crownY = p.topBack + 0.16f * len;
    eh.headHalfW = p.rxMax * 0.82f;
    const float wig = 0.05f * std::sin((float)head.phase * 0.11f) +
                      0.34f * head.expr.browUp - 0.30f * head.expr.browDown -
                      0.30f * head.expr.sad; // a sad animal's ears go back
    out.push_back(buildEar(-1.f, wig, eh, st));
    out.push_back(buildEar(+1.f, wig, eh, st));

    // Big eyes, well up the face and on the front of the cranium so both of
    // them read from straight on.
    {
        Mesh eyes;
        eyes.doubleSided = true;
        eyes.ambient = 0.26f;
        // Modest gloss: the bead is the lid as well as the eyeball, and at a
        // wet eyeball's shine a closed lid comes out looking like a jewel.
        eyes.spec = 0.34f;
        eyes.shin = 24;
        const float R = 0.21f * p.rxMax;
        for (float side : {-1.f, 1.f})
            addEyeAt(eyes, p, 0.52f, head.crownY + 0.26f * len, R, side, 0.85f,
                     Vec3f(18, 16, 20));
        eyes.computeNormals();
        out.push_back(std::move(eyes));
    }

    // The nose, a button on the front of the muzzle's point.
    Style ns = st;
    ns.noseR = 0.20f * p.rxMax;
    Head nh = head;
    nh.noseY = head.crownY + 0.97f * len;
    nh.noseZ = p.zTip + 0.03f;
    out.push_back(buildNose(nh, ns));
    // A button: carried by the muzzle, not stretched by it.
    out.back().pin(0, Vec3f(0.f, nh.noseY, nh.noseZ));

    // Buck teeth, from just under the nose and a shade in front of the
    // muzzle's face, so they show with the mouth shut as well as open --
    // which is the whole point of them. Set behind that face they were
    // simply swallowed by the closed jaw.
    const float tlen = (0.11f + 0.10f * head.expr.jawOpen) * len;
    out.push_back(buildIncisors(
        Vec3f(0.f, head.crownY + 1.03f * len, p.zTip - 0.04f),
        tlen, 0.075f * p.rxMax, 0.05f * p.rxMax, Vec3f(226, 238, 245)));

    // Cheek pouches. Out on the cheeks and flattened against them, in a
    // lighter chestnut than the coat so they read as the head bulging rather
    // than as two pale objects stuck on it. Absent until actually puffed.
    // A smile rounds the cheeks out a little, over and above any actual
    // puffing: on this head it is the clearest place a smile can show.
    const float puff = clampf(std::max(head.expr.cheekPuff,
                                       0.40f * head.expr.smile), 0.f, 1.f);
    if (puff > 0.04f) {
        const Section cs = sectionAt(0.50f, p);
        const float r = (0.16f + 0.26f * puff) * p.rxMax;
        for (float side : {-1.f, 1.f}) {
            const Vec3f c(side * (cs.rx * 0.74f + 0.10f * r),
                          head.crownY + 0.80f * len, cs.z - 0.10f);
            out.push_back(buildLobe(c, r, r * 0.84f, r * 0.55f,
                                    Vec3f(120, 168, 214), 0.50f, 0.16f, 16));
        }
    }
}

// Everything an elephant has that the loft does not: the ears, which are
// nearly the size of the head; the trunk, which curls up as the mouth opens;
// a pair of tusks flanking it; and small eyes set low and wide.
void addElephantTrim(std::vector<Mesh>& out, const Head& head,
                     const HeadShape& p) {
    const Style st = styleFor(Species::Elephant);
    const float len = head.chinY - head.crownY;

    // Ears, on the sides of the model's own skull rather than the real head's.
    Head eh = head;
    // Anchored at the side of the head at about eye level, not up on the
    // crown: hung from the crown the fans splay upward like wings.
    eh.crownY = p.topBack + 0.47f * len;
    eh.headHalfW = p.rxMax;
    // The mouth is behind the trunk and barely visible, so a smile and a sad
    // face have to show in the ears as well or they do not show at all.
    const float wig = 0.04f * std::sin((float)head.phase * 0.09f) +
                      0.22f * head.expr.browUp - 0.18f * head.expr.browDown +
                      0.20f * head.expr.smile - 0.24f * head.expr.sad;
    out.push_back(buildEar(-1.f, wig, eh, st));
    out.push_back(buildEar(+1.f, wig, eh, st));

    // Small eyes, low and wide -- an elephant's are tiny for the size of its
    // head, and putting big ones on it makes it a cartoon mouse.
    {
        Mesh eyes;
        eyes.doubleSided = true;
        eyes.ambient = 0.28f;
        eyes.spec = 0.32f;
        eyes.shin = 24;
        // Well forward on the head, not level with it: at mid-depth the
        // solved position lands exactly where the ear attaches and the eyes
        // are buried behind the fans. Further forward the head is narrower,
        // so they sit inboard of the ear roots and in front of them.
        const float R = 0.10f * p.rxMax;
        for (float side : {-1.f, 1.f})
            addEyeAt(eyes, p, 0.74f, head.crownY + 0.42f * len, R, side, 0.62f,
                     Vec3f(38, 34, 34));
        eyes.computeNormals();
        out.push_back(std::move(eyes));
    }

    // The trunk. It hangs from the middle of the face with a slight forward
    // bow, and opening the mouth raises and curls it -- an elephant about to
    // trumpet. Puckering curls just the tip, which is the bit that actually
    // moves most on a real one.
    const float raise = clampf(head.expr.jawOpen, 0.f, 1.f);
    const float tipCurl = clampf(head.expr.pucker, 0.f, 1.f);
    const Vec3f trunkBase(0.f, head.crownY + 0.52f * len, p.zTip + 0.18f);
    out.push_back(buildTaperTube(trunkBase,
                                 0.18f,                       // hangs down
                                 // Past a half turn the tip is not just
                                 // rising but coming back over, which is what
                                 // makes the raise read head-on instead of
                                 // merely foreshortening the trunk.
                                 // A smile curls the tip up too, and a sad
                                 // face lets it hang.
                                 0.26f + 2.70f * raise + 0.60f * tipCurl +
                                     0.55f * head.expr.smile -
                                     0.18f * head.expr.sad,
                                 0.f,
                                 (1.38f + 0.12f * raise) * len,
                                 0.29f * p.rxMax, 0.34f,
                                 Vec3f(128, 129, 134), 0.035f,
                                 0.44f, 0.16f, 14));

    // Tusks, flanking the trunk: shorter, splayed outward, curling forward
    // and up, and ivory rather than grey.
    for (float side : {-1.f, 1.f}) {
        const Vec3f tb(side * 0.46f * p.rxMax, head.crownY + 0.72f * len,
                       p.zTip + 0.14f);
        out.push_back(buildTaperTube(tb, 0.22f, 2.25f, side * 0.42f,
                                     0.95f * len, 0.115f * p.rxMax, 0.10f,
                                     Vec3f(214, 226, 234), 0.f,
                                     0.58f, 0.34f, 26));
    }
}

// Everything a T. rex has that the loft does not: amber slit-pupilled eyes
// under heavy brow horns, a row of spikes down the middle of the skull, and
// nostrils on top of the snout. The teeth are the shark's, fewer and bigger.
void addDinosaurTrim(std::vector<Mesh>& out, const Head& head,
                     const HeadShape& p) {
    const float len = head.chinY - head.crownY;
    const float sEye = 0.46f;
    const float yEye = head.crownY + 0.30f * len;

    // Eyes, set high on the sides of the skull and turned well forward so
    // both read from the front. A sad face half-lids them (in addEyeBead).
    {
        Mesh eyes;
        eyes.doubleSided = true;
        eyes.ambient = 0.30f;
        eyes.spec = 0.42f;
        eyes.shin = 30;
        const float R = 0.15f * p.rxMax;
        const Vec3f amber(40, 170, 236); // BGR
        for (float side : {-1.f, 1.f})
            addEyeAt(eyes, p, sEye, yEye, R, side, 2.00f, Vec3f(16, 18, 14),
                     &amber);
        eyes.computeNormals();
        out.push_back(std::move(eyes));
    }

    // Brow horns: a short bony horn over each eye, leaning out. Raising the
    // brows stands them up straight; a scowl splays them out and down over
    // the eyes, as does a sad face. Swung across the screen rather than
    // toward it -- head-on, a horn tipped forward does not visibly move.
    {
        const Section c = sectionAt(sEye, p);
        const float splay = 0.55f - 0.45f * head.expr.browUp +
                            0.95f * head.expr.browDown + 0.60f * head.expr.sad;
        const float yH = yEye - 0.16f * len;
        for (float side : {-1.f, 1.f}) {
            const float dy = c.cy - yH;
            const float n = clampf(dy / std::max(1e-3f, c.ryUp), -0.97f, 0.97f);
            const float fx = std::sqrt(std::max(0.f, 1.f - n * n));
            const Vec3f base(side * c.rx * fx * 0.92f, yH, c.z - 0.05f);
            const Vec3f dir(side * splay, -1.f, -0.30f);
            out.push_back(buildCone(base, dir, Vec3f(0.f, 0.f, 1.f),
                                    0.20f * len, 0.085f * p.rxMax,
                                    0.085f * p.rxMax, 0.35f,
                                    Vec3f(70, 128, 120), Vec3f(176, 210, 222)));
        }
    }

    // Spikes down the middle of the skull, biggest over the crown and
    // shrinking toward the snout: from the front they stand up out of the
    // top of the head as a crest, which is the silhouette that says dinosaur.
    // Thin plates rather than pegs, leaned back.
    {
        const int nSpike = 6;
        for (int i = 0; i < nSpike; ++i) {
            const float sS = 0.04f + 0.50f * (float)i / (nSpike - 1);
            const Section c = sectionAt(sS, p);
            const float k = 1.f - 0.55f * (float)i / (nSpike - 1);
            const Vec3f base(0.f, c.cy - c.ryUp, c.z);
            out.push_back(buildCone(base, Vec3f(0.f, -1.f, 0.10f),
                                    Vec3f(0.f, 0.f, 1.f), 0.24f * len * k,
                                    0.075f * p.rxMax * k, 0.15f * p.rxMax * k,
                                    0.45f, Vec3f(44, 112, 200),
                                    Vec3f(70, 190, 250)));
        }
    }

    // Nostrils: two dark pits on top of the front of the snout, facing up and
    // forward.
    {
        // Just behind where the dome starts: any further forward and the
        // shrinking cross-sections swallow them.
        const Section c = sectionAt(0.80f, p);
        for (float side : {-1.f, 1.f}) {
            const Vec3f ctr(side * 0.36f * c.rx, c.cy - 0.86f * c.ryUp,
                            c.z - 0.08f);
            out.push_back(buildLobe(ctr, 0.17f * c.rx, 0.11f * c.rx,
                                    0.05f * c.rx, Vec3f(22, 42, 28),
                                    0.40f, 0.10f, 10));
        }
    }
}

// Everything a dragon has that the loft does not: gold slit-pupilled eyes,
// horns sweeping up and back off the skull, finned frills at the sides, a
// crest of spikes, spikes at the corners of the jaw and nostrils on the snout.
void addDragonTrim(std::vector<Mesh>& out, const Head& head,
                   const HeadShape& p) {
    const float len = head.chinY - head.crownY;
    const float sEye = 0.44f;
    const float yEye = head.crownY + 0.30f * len;

    {
        Mesh eyes;
        eyes.doubleSided = true;
        eyes.ambient = 0.34f;
        eyes.spec = 0.48f;
        eyes.shin = 32;
        const float R = 0.15f * p.rxMax;
        const Vec3f gold(20, 205, 255); // BGR
        for (float side : {-1.f, 1.f})
            addEyeAt(eyes, p, sEye, yEye, R, side, 2.00f, Vec3f(10, 12, 16),
                     &gold);
        eyes.computeNormals();
        out.push_back(std::move(eyes));
    }

    // Horns: from the top of the skull behind the eyes, rising and curling
    // back. Curled only part of the way: head-on, a horn swept straight back
    // is end-on and disappears behind the head that grows it.
    {
        const Section c = sectionAt(0.20f, p);
        for (float side : {-1.f, 1.f}) {
            const Vec3f base(side * 0.52f * c.rx, c.cy - 0.82f * c.ryUp, c.z);
            out.push_back(buildTaperTube(base, 0.88f * kPi, 1.30f * kPi,
                                         side * 0.78f, 0.82f * len,
                                         0.12f * p.rxMax, 0.08f,
                                         Vec3f(130, 176, 204), 0.05f,
                                         0.50f, 0.30f, 22));
        }
    }

    // Frills, on the sides of the model's skull, raised and laid back by the
    // brows like the animals' ears.
    {
        const Style st = styleFor(Species::Dragon);
        Head eh = head;
        eh.crownY = p.topBack + 0.36f * len;
        eh.headHalfW = p.rxMax * 0.92f;
        const float wig = 0.05f * std::sin((float)head.phase * 0.11f) +
                          0.30f * head.expr.browUp - 0.26f * head.expr.browDown -
                          0.20f * head.expr.sad;
        out.push_back(buildEar(-1.f, wig, eh, st));
        out.push_back(buildEar(+1.f, wig, eh, st));
    }

    // A crest down the middle of the skull, tallest over the crown.
    {
        const int nSpike = 7;
        for (int i = 0; i < nSpike; ++i) {
            const float sS = 0.03f + 0.58f * (float)i / (nSpike - 1);
            const Section c = sectionAt(sS, p);
            const float k = 1.f - 0.60f * (float)i / (nSpike - 1);
            const Vec3f base(0.f, c.cy - c.ryUp, c.z);
            out.push_back(buildCone(base, Vec3f(0.f, -1.f, 0.12f),
                                    Vec3f(0.f, 0.f, 1.f), 0.20f * len * k,
                                    0.06f * p.rxMax * k, 0.14f * p.rxMax * k,
                                    0.50f, Vec3f(40, 60, 170),
                                    Vec3f(60, 170, 245)));
        }
    }

    // Spikes at the corners of the jaw, pointing out and back: they put the
    // dragon's jawline on the silhouette.
    {
        const Section c = sectionAt(p.sHinge, p);
        for (float side : {-1.f, 1.f}) {
            const Vec3f base(side * 0.96f * c.rx, c.cy + 0.25f * c.ryLo, c.z);
            out.push_back(buildCone(base, Vec3f(side * 1.f, 0.25f, 0.35f),
                                    Vec3f(0.f, 0.f, 1.f), 0.20f * len,
                                    0.06f * p.rxMax, 0.06f * p.rxMax, 0.30f,
                                    Vec3f(40, 60, 170), Vec3f(150, 192, 214)));
        }
    }

    // Nostrils, on top of the end of the snout.
    {
        const Section c = sectionAt(0.82f, p);
        for (float side : {-1.f, 1.f}) {
            const Vec3f ctr(side * 0.34f * c.rx, c.cy - 0.86f * c.ryUp,
                            c.z - 0.08f);
            out.push_back(buildLobe(ctr, 0.15f * c.rx, 0.10f * c.rx,
                                    0.05f * c.rx, Vec3f(16, 16, 50),
                                    0.40f, 0.10f, 10));
        }
    }
}

std::vector<Mesh> buildMeshes(const Head& head, Species species,
                              Fire* fire = nullptr) {

    std::vector<Mesh> meshes;
    if (fire) *fire = Fire{};
    if (species == Species::Shark || species == Species::Squirrel ||
        species == Species::Elephant || species == Species::Dinosaur ||
        species == Species::Dragon) {
        const bool shark = (species == Species::Shark);
        const bool dragon = (species == Species::Dragon);
        // The dragon is built on the dinosaur: the same loft, toothed jaw and
        // hinge, with its own outline and trim.
        const bool dino = (species == Species::Dinosaur) || dragon;
        const HeadShape sp = shark      ? sharkShape(head)
                             : dragon   ? dragonShape(head)
                             : dino     ? dinosaurShape(head)
                             : (species == Species::Squirrel)
                                 ? squirrelShape(head)
                                 : elephantShape(head);
        // Everything is built at rest first, then has the face's motion
        // mapped onto it, and only then is the jaw swung: the mapping pairs
        // the model's features with the face's as they sit at rest.
        meshes.push_back(buildHeadHalf(sp, true));
        Mesh jaw = buildHeadHalf(sp, false);
        Mesh tongue = buildMuzzleTongue(sp, dino ? 0.92f : 10.f);
        jaw.jaw = tongue.jaw = true;
        // The jaw and everything in it are one group: they hinge together,
        // and slide together when the jaw is worked to one side.
        std::vector<Mesh*> group{&jaw, &tongue};
        Mesh upTeeth, lowTeeth;
        const bool teeth = shark || dino;
        if (shark) {
            upTeeth = buildSharkTeeth(sp, true);
            lowTeeth = buildSharkTeeth(sp, false);
        } else if (dragon) {
            // Fewer and longer: fangs.
            const Vec3f enamel(220, 238, 244), root(150, 180, 196);
            upTeeth = buildSharkTeeth(sp, true, 7, 1.50f, enamel, root);
            lowTeeth = buildSharkTeeth(sp, false, 6, 1.30f, enamel, root);
        } else if (dino) {
            const Vec3f enamel(214, 236, 242), root(150, 186, 200);
            upTeeth = buildSharkTeeth(sp, true, 8, 1.35f, enamel, root);
            lowTeeth = buildSharkTeeth(sp, false, 7, 1.15f, enamel, root);
        }
        lowTeeth.jaw = true;
        if (teeth) {
            meshes.push_back(std::move(upTeeth));
            group.push_back(&lowTeeth);
        }
        // Where the eyes are: the trim builders place them, and the mapping
        // pairs them with the face's.
        const float len = head.chinY - head.crownY;
        Vec3f eye;
        if (shark) {
            // The same parameters the eyes had when they lived inside
            // buildSharkTrim, before the other two needed them too.
            meshes.push_back(buildEyeBeads(sp, 0.34f, 0.52f, 0.140f * sp.rxMax,
                                           Vec3f(16, 15, 18), 0.30f, 0.34f, 26));
            meshes.push_back(buildSharkTrim(sp));
            const Section ce = sectionAt(0.34f, sp);
            eye = Vec3f(ce.rx * std::cos(0.52f) * 0.99f,
                        ce.cy - ce.ryUp * std::sin(0.52f) * 0.99f, ce.z);
        } else if (dragon) {
            addDragonTrim(meshes, head, sp);
            eye = eyeCentreAt(sp, 0.44f, head.crownY + 0.30f * len, 1.f);
        } else if (dino) {
            addDinosaurTrim(meshes, head, sp);
            eye = eyeCentreAt(sp, 0.46f, head.crownY + 0.30f * len, 1.f);
        } else if (species == Species::Squirrel) {
            addSquirrelTrim(meshes, head, sp);
            eye = eyeCentreAt(sp, 0.52f, head.crownY + 0.26f * len, 1.f);
        } else {
            addElephantTrim(meshes, head, sp);
            eye = eyeCentreAt(sp, 0.74f, head.crownY + 0.42f * len, 1.f);
        }

        // The face's motion, point by point. Exaggerated a little: these
        // heads are cartoons, and a cartoon's expressions are bigger than
        // the face making them.
        const bool meshDriven = mapped(head);
        if (meshDriven) {
            std::vector<const Mesh*> all;
            for (const Mesh& m : meshes) all.push_back(&m);
            for (const Mesh* m : group) all.push_back(m);
            const MotionMap mm = motionMap(faceMotion(head, true),
                                           loftAnchors(head, sp, eye), all,
                                           kLoftGain);
            for (Mesh& m : meshes) deform(m, mm);
            for (Mesh* m : group) deform(*m, mm);
        }

        // A little always ajar, so the teeth show even with the mouth shut,
        // then opened the rest of the way by the jawOpen blendshape.
        // How far the jaw drops. A shark's gape is enormous and the others
        // are not, but all three were opening far too little to read as an
        // open mouth at all.
        // The dinosaur sits between: a big gape, but a heavier jaw.
        const float ang = (shark ? 0.07f : dino ? 0.05f : 0.04f) +
                          (shark ? 0.78f : dino ? 0.66f : 0.52f) *
                              clampf(head.expr.jawOpen, 0.f, 1.f);
        const Section hinge = sectionAt(sp.sHinge, sp);
        const Vec3f pivot(0.f, hinge.cy, hinge.z);
        // +x is the head's right, so a positive rotation about it swings
        // what is in front of the hinge downward -- the jaw dropping open.
        const Matx33f Rj = rotAxis(Vec3f(1.f, 0.f, 0.f), ang);
        // Worked to one side. The mapped motion already carries the jaw's
        // sideways shift, so the blendshape only drives it without a mesh.
        const float slide =
            meshDriven ? 0.f : 0.22f * head.expr.jawSide * sp.rxMax;
        // Behind the hinge the jaw is a whole cross-section of the head, not
        // half of one (see buildHeadHalf), so swinging it rigidly lifts a
        // skull-sized cap up over the top of the head. On the dinosaur, whose
        // hinge is far back, that cap covered the eyes whenever the mouth
        // opened, and so it did on the squirrel and the elephant once their
        // heads were sized from the face at rest, which is longer in the jaw
        // than the open-mouthed one they used to be measured from. So for
        // all but the shark, whose hinge is right at the back, the swing
        // fades in across the hinge instead: the back of the jaw stays put
        // and only what is in front of it drops.
        const float ramp = 0.30f;
        for (Mesh* j : group) {
            for (auto& q : j->pos) {
                if (!shark) {
                    // The sideways slide fades in the same way, or the back
                    // of the jaw slides out past the skull as a slab.
                    float w = clampf((pivot[2] - q[2]) / ramp, 0.f, 1.f);
                    w = w * w * (3.f - 2.f * w);
                    // Where the jaw stays put it is the very surface of the
                    // skull -- behind the hinge both are whole ellipses --
                    // and the two z-fight into specks on top of the head.
                    // Tucked a hair inside, the skull wins cleanly.
                    const float tuck = 1.f - 0.015f * (1.f - w);
                    q[0] *= tuck;
                    q[1] = pivot[1] + (q[1] - pivot[1]) * tuck;
                    q = rotAbout(rotAxis(Vec3f(1.f, 0.f, 0.f), ang * w), q, pivot);
                    q[0] += slide * w;
                } else {
                    q = rotAbout(Rj, q, pivot);
                    q[0] += slide;
                }
            }
            j->computeNormals();
        }
        meshes.push_back(std::move(jaw));
        if (teeth) meshes.push_back(std::move(lowTeeth));
        meshes.push_back(std::move(tongue));

        // Fire, once the mouth is open far enough to breathe it -- not at the
        // first parting of the lips, which happens talking.
        if (dragon && fire) {
            const float amt = smooth01(0.30f, 0.70f, head.expr.jawOpen);
            if (amt > 0.f) {
                // Between the jaws near the front: the upper mouth line, and
                // the same point on the lower jaw swung open with it.
                const Section c = sectionAt(0.90f, sp);
                const Vec3f up(0.f, c.cy, c.z);
                const Vec3f lo = rotAbout(rotAxis(Vec3f(1.f, 0.f, 0.f), ang),
                                          up, pivot);
                fire->amount = amt;
                fire->origin = (up + lo) * 0.5f;
                // Along the middle of the gape, tipped a little further down
                // so it visibly pours out of the mouth. Mostly at the camera,
                // though: head-on that is a fireball swelling out of the jaws
                // toward you, which is the picture; aimed down it mostly
                // leaves the bottom of the frame.
                const float a = 0.5f * ang + 0.15f;
                fire->dir = Vec3f(0.f, std::sin(a), -std::cos(a));
                fire->length = (0.55f + 0.45f * amt) * 1.45f * len;
            }
        }
    } else {
        const Style st = styleFor(species);
        // Ears answer the brows. Raising them pricks the ears up and out,
        // lowering them lays them back -- the same thing the animal would do,
        // and the clearest way for a rigid part to show an expression that
        // otherwise only the painted face carries.
        const float wig = 0.05f * std::sin((float)head.phase * 0.11f) +
                          0.34f * head.expr.browUp - 0.30f * head.expr.browDown;
        meshes.push_back(buildEar(-1.f, wig, head, st));
        meshes.push_back(buildEar(+1.f, wig, head, st));
        if (st.snout) {
            meshes.push_back(buildSnout(head, st));
        } else {
            meshes.push_back(buildNose(head, st));
        }
        // A tongue, out of the mouth. Driven by tongueOut where the model
        // scores it, and otherwise by the jaw simply being open -- which is
        // both true of a real mouth and the only thing that reliably fires,
        // since MediaPipe's tongueOut seldom rises above its noise floor.
        const float tOut = std::max(head.expr.tongue, 0.55f * head.expr.jawOpen);
        if (tOut > 0.02f) {
            const float drop = head.chinY - head.noseY;
            const float mouthY = head.noseY + 0.42f * drop;
            const Vec3f base(0.f, mouthY, head.noseZ * 0.55f);
            const float L = (0.35f + 0.85f * tOut) * drop;
            const Vec3f tip = base + Vec3f(0.f, L, -0.34f * L);
            meshes.push_back(buildTongue(base, tip, 0.27f * head.headHalfW,
                                         0.05f * head.headHalfW, -0.22f,
                                         st.tongueCol));
            meshes.back().jaw = true;
        }
        // These parts sit on the face itself, so each vertex follows the
        // face exactly where it is: no warp, and no exaggeration, or the nose
        // would slide off the painted muzzle under it.
        if (mapped(head)) {
            std::vector<const Mesh*> all;
            for (const Mesh& m : meshes) all.push_back(&m);
            const MotionMap mm =
                motionMap(faceMotion(head, false), Anchors{}, all, 1.f);
            for (Mesh& m : meshes) deform(m, mm);
        }
    }

    return meshes;
}

// Image-space bounding box of every projected vertex -> the region touched.
cv::Rect meshBounds(const std::vector<Mesh>& meshes, const Head& head,
                    int w, int h) {
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    for (const Mesh& m : meshes)
        for (const Vec3f& p : m.pos) {
            const Proj pr = project(head, head.R * (p * head.unit), 1.f,
                                    cv::Point2f(0, 0));
            minx = std::min(minx, pr.s.x); maxx = std::max(maxx, pr.s.x);
            miny = std::min(miny, pr.s.y); maxy = std::max(maxy, pr.s.y);
        }
    cv::Rect roi((int)std::floor(minx) - 2, (int)std::floor(miny) - 2,
                 (int)std::ceil(maxx - minx) + 4, (int)std::ceil(maxy - miny) + 4);
    return roi & cv::Rect(0, 0, w, h);
}

// The model for `head`, built once per frame however often it is asked for.
//
// The preview asks twice -- bounds() to size the region it converts, then
// render() to draw into it -- with the same face, and building the model is
// most of the cost that is not rasterising: the lofts, and since the mapping,
// the face's motion field and the warp. Nothing that goes into the model
// depends on where it lands on screen or how big it is (R, unit and anchor
// only enter at projection), so those are left out of the comparison, which
// also lets the two calls differ by the region origin the way they do.
// render() is called from the one preview path, as the scratch buffers
// below already assume, so a single entry is enough.
const std::vector<Mesh>& modelFor(const Head& head, Species species,
                                  Fire* fire = nullptr) {
    struct Entry {
        bool valid = false;
        Species species = Species::Dog;
        float shape[5] = {};
        Expression expr;
        double phase = 0.0;
        std::vector<Vec3f> live, rest;
        std::vector<Mesh> meshes;
        Fire fire;
    };
    static Entry cache;
    const float shape[5] = {head.headHalfW, head.crownY, head.noseY,
                            head.noseZ, head.chinY};
    auto sameVec = [](const std::vector<Vec3f>& a, const std::vector<Vec3f>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            for (int k = 0; k < 3; ++k)
                if (a[i][k] != b[i][k]) return false;
        return true;
    };
    const Expression& e = head.expr;
    const Expression& c = cache.expr;
    const bool hit =
        cache.valid && cache.species == species &&
        std::equal(shape, shape + 5, cache.shape) && cache.phase == head.phase &&
        e.jawOpen == c.jawOpen && e.smile == c.smile && e.frown == c.frown &&
        e.sad == c.sad && e.blinkL == c.blinkL && e.blinkR == c.blinkR &&
        e.browUp == c.browUp && e.browDown == c.browDown &&
        e.pucker == c.pucker && e.cheekPuff == c.cheekPuff &&
        e.tongue == c.tongue && e.jawSide == c.jawSide &&
        sameVec(head.live, cache.live) && sameVec(head.rest, cache.rest);
    if (!hit) {
        cache.meshes = buildMeshes(head, species, &cache.fire);
        cache.valid = true;
        cache.species = species;
        std::copy(shape, shape + 5, cache.shape);
        cache.expr = head.expr;
        cache.phase = head.phase;
        cache.live = head.live;
        cache.rest = head.rest;
    }
    if (fire) *fire = cache.fire;
    return cache.meshes;
}

cv::Rect bounds(const Head& head, Species species) {
    if (head.unit < 12.f) return cv::Rect();
    // A frame-sized clip, because the caller intersects with the frame itself.
    const int big = 1 << 20;
    Fire fire;
    const cv::Rect model = meshBounds(modelFor(head, species, &fire), head, big, big);
    const cv::Rect flames = fireBounds(head, fire);
    if (flames.area() == 0) return model;
    return model.area() == 0 ? flames : (model | flames);
}

void render(cv::Mat& frame, const Head& head, Species species) {
    if (frame.empty() || frame.type() != CV_8UC3) return;
    if (head.unit < 12.f) return; // too small to render cleanly

    Fire fire;
    const std::vector<Mesh>& meshes = modelFor(head, species, &fire);
    const cv::Rect roi = meshBounds(meshes, head, frame.cols, frame.rows);
    if (roi.width < 2 || roi.height < 2) return;

    const cv::Point2f org((float)roi.x, (float)roi.y);
    const float ss = clampf(std::sqrt((float)kMaxSamples /
                                      (float)std::max(1, roi.width * roi.height)),
                            1.f, (float)kSS);
    const int LW = (int)std::lround(roi.width * ss);
    const int LH = (int)std::lround(roi.height * ss);
    // Kept between frames rather than reallocated: at kSS = 2 these are tens
    // of megabytes, and churning that every frame costs more in allocation
    // and page faults than the drawing does. Reused only when the size still
    // fits, and always cleared. render() is called from the one preview path,
    // so these are not shared across threads.
    static cv::Mat layer, cover, zbuf;
    if (layer.rows < LH || layer.cols < LW) {
        layer.create(LH, LW, CV_32FC3);
        cover.create(LH, LW, CV_32F);
        zbuf.create(LH, LW, CV_32F);
    }
    cv::Mat lay = layer(cv::Rect(0, 0, LW, LH));
    cv::Mat cov = cover(cv::Rect(0, 0, LW, LH));
    cv::Mat zbf = zbuf(cv::Rect(0, 0, LW, LH));
    lay.setTo(cv::Scalar(0, 0, 0));
    cov.setTo(cv::Scalar(0));
    zbf.setTo(cv::Scalar(1e9f));
    for (const Mesh& m : meshes)
        raster(m, head, ss, org, lay, cov, zbf);

    cv::Mat colDown, covDown;
    cv::resize(lay, colDown, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_AREA);
    cv::resize(cov, covDown, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_AREA);

    cv::Mat dst = frame(roi);
    for (int y = 0; y < roi.height; ++y) {
        cv::Vec3b* d = dst.ptr<cv::Vec3b>(y);
        const cv::Vec3f* c = colDown.ptr<cv::Vec3f>(y);
        const float* a = covDown.ptr<float>(y);
        for (int x = 0; x < roi.width; ++x) {
            const float cov = a[x];
            if (cov <= 0.003f) continue;
            // colDown is the coverage-weighted colour sum from INTER_AREA, so
            // the lit colour is colDown/cover; composite that over by cov.
            for (int k = 0; k < 3; ++k) {
                const float src = c[x][k] / cov;
                d[x][k] = cv::saturate_cast<uchar>(src * cov + d[x][k] * (1.f - cov));
            }
        }
    }
    // Over everything: the jet comes out toward the camera, so nothing of the
    // model is in front of it.
    drawFire(frame, head, fire);
}

} // namespace olc::face3d
