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

    int add(const Vec3f& p, const Vec3f& c) {
        pos.push_back(p);
        col.push_back(c);
        nrm.emplace_back(0.f, 0.f, 0.f);
        return (int)pos.size() - 1;
    }
    void face(int a, int b, int c) { tri.emplace_back(a, b, c); }

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
struct SharkProfile {
    // The head's longitudinal axis runs from s = 0 at the back of the skull to
    // s = 1 at the point of the snout.
    //
    // The silhouette is given as three curves rather than one axis plus a
    // radius, because head-on the outline is what has to be got right and an
    // axis-plus-radius body cannot produce it: whatever tapers forward hides
    // inside the largest cross-section, which is the one that has to cover
    // the whole head. So the snout is driven *past* the chin instead, where
    // it is actually seen.
    float crown, chin;     // the real head, which everything is sized to
    float topBack;         // top of the skull, a little clear of the crown
    float cyBack, cyTip;   // the mouth line, at the corners and at the point
    float botBack;         // bottom of the jaw, at the back
    float rxMax;
    float sHinge;          // where the jaw parts company with the skull
    float zBack, zTip;
    Expression e;          // what the face wearing it is doing
};

SharkProfile sharkProfile(const Head& h) {
    SharkProfile p;
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
    p.cyBack = h.crownY + (0.56f - 0.14f * h.expr.smile + 0.08f * h.expr.frown) * len;
    p.cyTip = h.chinY + 0.34f * len;
    p.botBack = h.chinY + 0.12f * len;
    p.rxMax = h.headHalfW * 1.40f;
    // Clearance over the crown, so the top of a real head stays inside the
    // shell rather than poking through the back of it.
    p.topBack = h.crownY - 0.17f * len;
    p.sHinge = 0.19f;
    p.zBack = 1.20f;
    p.zTip = h.noseZ - 1.15f;
    return p;
}

// Girth along the head: held near full through the braincase, which has a
// whole head to cover, then drawn away hard into the point of the snout.
float sharkGirth(float s) {
    const float t = clampf(s, 0.f, 1.f);
    const float taper = std::pow(std::max(0.f, 1.f - std::pow(t, 1.75f)), 0.60f);
    return taper * (1.f + 0.08f * std::sin(kPi * std::pow(t, 0.8f)));
}

// How far the jaw has parted from the skull at station s: 0 behind the hinge,
// where the two are one closed head, 1 in front of it, where the skull's
// cross-section is the upper half and the jaw's the lower. Blended over a
// short run so neither develops a crease at the hinge.
float sharkGape(float s, const SharkProfile& p) {
    return clampf((s - p.sHinge) / 0.11f, 0.f, 1.f);
}

struct SharkSection {
    float z, cy, rx, ryUp, ryLo, gape;
};

SharkSection sharkSection(float s, const SharkProfile& p) {
    SharkSection c;
    const float t = clampf(s, 0.f, 1.f);
    c.z = p.zBack + (p.zTip - p.zBack) * t;
    // Top of the skull: holds the crown over the braincase, then dives along
    // the rostrum to meet the mouth line at the point.
    const float yTop = p.topBack + (p.cyTip - p.topBack) * std::pow(t, 1.75f);
    c.cy = p.cyBack + (p.cyTip - p.cyBack) * std::pow(t, 1.45f);
    const float yBot = p.botBack + (p.cyTip - p.botBack) * std::pow(t, 1.15f);
    c.ryUp = std::max(0.f, c.cy - yTop);
    c.ryLo = std::max(0.f, yBot - c.cy);
    c.rx = p.rxMax * sharkGirth(t);
    c.gape = sharkGape(t, p);
    return c;
}

// Countershading: dark along the back, abruptly white underneath, which is
// the one marking that makes a grey shape read as a shark.
// Where a point sits down the model: 0 at the top of the skull, 1 at the
// point of the snout, which is its lowest part.
float sharkDepth(float y, const SharkProfile& p) {
    return clampf((y - p.topBack) / std::max(1e-3f, p.cyTip - p.topBack),
                  0.f, 1.f);
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
Vec3f sharkSkin(float v, float s, float ax) {
    const Vec3f back(112, 108, 104);   // BGR: slate grey
    const Vec3f flank(150, 148, 146);
    const Vec3f belly(228, 231, 234);
    const float t = clampf((v - 0.36f) * 7.0f, 0.f, 1.f);
    Vec3f c = back * (1.f - t) + flank * t;
    const float b = clampf((v - 0.53f) * 7.4f, 0.f, 1.f);
    c = c * (1.f - b) + belly * b;
    // Gill slits: five short dark bars on the flanks, behind the mouth corner
    // and never on the belly, where a shark has none.
    if (s > 0.05f && s < 0.24f && ax > 0.26f && v > 0.26f && v < 0.54f) {
        const float ph = (s - 0.045f) / 0.032f;       // one bar per 0.032 of s
        const float d = std::fabs(ph - std::floor(ph) - 0.5f) * 2.f;
        const float slit = clampf((0.42f - d) * 5.f, 0.f, 1.f) *
                           clampf((ax - 0.30f) * 3.f, 0.f, 1.f);
        c = c * (1.f - slit * 0.85f) + Vec3f(52, 50, 54) * (slit * 0.85f);
    }
    return c;
}

// One half of the head: the skull if `upper`, the lower jaw otherwise.
//
// Each cross-section is a closed outline -- the outer arc, then a return along
// the mouth line -- swept along the head's axis as a single uniform grid, so
// one triangle pattern winds the whole thing and the interior of the mouth
// closes itself. Behind the hinge the return bulges out into the other half of
// the ellipse instead of running flat, which is what makes the head solid
// there rather than two shells with a slot between them.
Mesh buildSharkHalf(const SharkProfile& p, bool upper) {
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
        const SharkSection c = sharkSection(s, p);
        const float rOut = upper ? c.ryUp : c.ryLo;
        // The return: the far half of the ellipse behind the hinge, flattening
        // into the roof or floor of the mouth in front of it.
        const float rRet = (upper ? c.ryLo : c.ryUp) * (1.f - c.gape);

        for (int k = 0; k < nArc; ++k) {
            const float th = kPi * (float)k / (nArc - 1); // +x round to -x
            const float ca = std::cos(th), sa = std::sin(th);
            const float yw = c.cy + dir * rOut * sa;
            ring[si].push_back(m.add(Vec3f(c.rx * ca, yw, c.z),
                                     sharkSkin(sharkDepth(yw, p), s,
                                               std::fabs(ca))));
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
            const Vec3f skin = sharkSkin(sharkDepth(yw, p), s, std::fabs(ca));
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
Mesh buildSharkTeeth(const SharkProfile& p, bool upper) {
    Mesh m;
    m.doubleSided = true; // far too small to be worth getting winding right
    m.ambient = 0.62f;
    m.spec = 0.30f;
    m.shin = 26;
    const Vec3f enamel(238, 243, 246);
    const Vec3f root(196, 206, 214);
    const float dir = upper ? 1.f : -1.f; // teeth point across the gape

    const int nTooth = 11;
    for (int i = 0; i < nTooth; ++i) {
        // Spread from just inside the mouth corner to near the snout's point.
        const float s = p.sHinge + 0.10f +
                        (0.94f - p.sHinge - 0.10f) * (float)i / (nTooth - 1);
        const SharkSection c = sharkSection(s, p);
        const float step = 0.030f;
        const SharkSection cA = sharkSection(std::max(0.f, s - step), p);
        const SharkSection cB = sharkSection(std::min(1.f, s + step), p);
        // Teeth shrink toward the point of the snout, as they do on a real jaw.
        const float len = (upper ? 0.26f : 0.22f) * (p.chin - p.cyBack) *
                          (0.55f + 0.45f * (1.f - s));
        const float half = 0.40f;

        for (float side : {-1.f, 1.f}) {
            const Vec3f a(side * (cA.rx * (1.f - half) + c.rx * half), cA.cy, cA.z);
            const Vec3f b(side * (cB.rx * (1.f - half) + c.rx * half), cB.cy, cB.z);
            const Vec3f inner(side * c.rx * 0.62f, c.cy, c.z);
            const Vec3f tipP(side * c.rx * 0.80f, c.cy + dir * len, c.z);
            const int ia = m.add(a, root), ib = m.add(b, root);
            const int ii = m.add(inner, root), it = m.add(tipP, enamel);
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
Mesh buildSharkTongue(const SharkProfile& p) {
    const float s0 = p.sHinge + 0.10f;
    const SharkSection a = sharkSection(s0, p);
    // Out past the teeth only as far as the tongue is actually out, and only
    // when there is a gap for it to come through.
    const float reach = 0.45f + 0.95f * p.e.tongue * (0.35f + 0.65f * p.e.jawOpen);
    const Vec3f base(0.f, a.cy, a.z);
    const Vec3f far(0.f, p.cyTip, p.zTip);
    const Vec3f tip = base + (far - base) * reach;
    return buildTongue(base, tip, 0.66f * a.rx, 0.10f * a.rx, 0.16f,
                       Vec3f(126, 112, 206));
}

Mesh buildSharkTrim(const SharkProfile& p) {
    Mesh m;
    m.doubleSided = true;
    m.ambient = 0.30f;
    m.spec = 0.55f;
    m.shin = 40;

    const float sEye = 0.34f;
    const SharkSection ce = sharkSection(sEye, p);
    const float thEye = 0.52f; // round from the flank toward the top
    for (float side : {-1.f, 1.f}) {
        const Vec3f centre(side * ce.rx * std::cos(thEye) * 0.99f,
                           ce.cy - ce.ryUp * std::sin(thEye) * 0.99f, ce.z);
        // Outward normal of the ellipse there, so the bead sits proud of the
        // surface however the head is proportioned.
        const Vec3f nOut = norm(Vec3f(side * std::cos(thEye) / ce.rx,
                                      -std::sin(thEye) / ce.ryUp, -0.25f));
        Vec3f u = norm(nOut.cross(Vec3f(0.f, 1.f, 0.f)));
        Vec3f v = norm(nOut.cross(u));
        const float R = 0.140f * p.rxMax;
        // Blinking. A shark has a nictitating membrane rather than a lid, and
        // either way a bead this small cannot show one: squashing the bead to
        // a slit is what reads as an eye closing at this size. `side` is -1
        // for the image-left eye, which is the one blinkL describes.
        const float blink = (side < 0.f) ? p.e.blinkL : p.e.blinkR;
        const float lid = std::max(0.07f, 1.f - blink);
        const int nSeg = 14, nRing = 6;
        std::vector<std::vector<int>> ring(nRing);
        for (int r = 0; r < nRing; ++r) {
            const float lat = 0.5f * kPi * (float)r / (nRing - 1);
            const float cr = std::cos(lat) * R, cz = std::sin(lat) * R;
            for (int i = 0; i < nSeg; ++i) {
                const float a = 2.f * kPi * i / nSeg;
                ring[r].push_back(m.add(centre + u * (cr * std::cos(a)) +
                                            v * (cr * std::sin(a) * lid) +
                                            nOut * (cz * lid),
                                        Vec3f(16, 15, 18)));
            }
        }
        for (int r = 0; r + 1 < nRing; ++r)
            for (int i = 0; i < nSeg; ++i) {
                const int j = (i + 1) % nSeg;
                m.face(ring[r][i], ring[r][j], ring[r + 1][j]);
                m.face(ring[r][i], ring[r + 1][j], ring[r + 1][i]);
            }
    }

    // Dorsal fin. Anatomically it belongs further down the animal, but it is
    // half of what makes a grey shape read as a shark at a glance, so it
    // stands on the back of the skull. A swept triangle with real chord and
    // thickness -- built as an outline given a left and a right copy, which
    // is why it needs no winding care.
    {
        const SharkSection a = sharkSection(0.06f, p);
        const SharkSection b = sharkSection(0.40f, p);
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

std::vector<Mesh> buildMeshes(const Head& head, Species species) {

    std::vector<Mesh> meshes;
    if (species == Species::Shark) {
        const SharkProfile sp = sharkProfile(head);
        // A little always ajar, so the teeth show even with the mouth shut,
        // then opened the rest of the way by the jawOpen blendshape.
        const float ang = 0.07f + 0.34f * clampf(head.expr.jawOpen, 0.f, 1.f);
        const SharkSection hinge = sharkSection(sp.sHinge, sp);
        const Vec3f pivot(0.f, hinge.cy, hinge.z);
        // +x is the head's right, so a positive rotation about it swings
        // what is in front of the hinge downward -- the jaw dropping open.
        const Matx33f Rj = rotAxis(Vec3f(1.f, 0.f, 0.f), ang);

        meshes.push_back(buildSharkHalf(sp, true));
        meshes.push_back(buildSharkTeeth(sp, true));
        Mesh jaw = buildSharkHalf(sp, false);
        Mesh lowTeeth = buildSharkTeeth(sp, false);
        Mesh tongue = buildSharkTongue(sp);
        // The jaw, its teeth and the tongue are one group: they hinge
        // together, and slide together when the jaw is worked to one side.
        const float slide = 0.22f * head.expr.jawSide * sp.rxMax;
        for (Mesh* j : {&jaw, &lowTeeth, &tongue}) {
            for (auto& q : j->pos) {
                q = rotAbout(Rj, q, pivot);
                q[0] += slide;
            }
            j->computeNormals();
        }
        meshes.push_back(std::move(jaw));
        meshes.push_back(std::move(lowTeeth));
        meshes.push_back(std::move(tongue));
        meshes.push_back(buildSharkTrim(sp));
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

cv::Rect bounds(const Head& head, Species species) {
    if (head.unit < 12.f) return cv::Rect();
    // A frame-sized clip, because the caller intersects with the frame itself.
    const int big = 1 << 20;
    return meshBounds(buildMeshes(head, species), head, big, big);
}

void render(cv::Mat& frame, const Head& head, Species species) {
    if (frame.empty() || frame.type() != CV_8UC3) return;
    if (head.unit < 12.f) return; // too small to render cleanly

    const std::vector<Mesh> meshes = buildMeshes(head, species);
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
}

} // namespace olc::face3d
