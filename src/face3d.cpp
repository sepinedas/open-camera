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

// --- Per-species geometry, in eye-separation units in the head frame -------
// +x toward the image-right eye, +y toward the chin, +z away from the camera,
// origin at the eye midpoint. Ear anchors are relative to the *measured* crown
// and head width, so they land on the real silhouette rather than an assumed
// one, and the muzzle sits on the measured nose.
struct Style {
    // Ear: a lobe swept from an attachment on the head out to a tip.
    float earAttachX, earAttachY;
    float earTipX, earTipY;
    float earHalfW;  // widest half-width, as a multiple of headHalfW
    float earRound;  // tip shape: small rounds it off, large draws it to a point
    float earBulge, earCurl;
    float earMottle; // per-vertex coat variation, so it is not moulded plastic
    Vec3f earCol, earInnerCol;

    // Muzzle. A dog gets a bare domed nose sitting on the painted muzzle; a
    // pig gets a snout -- a short tube standing off the face, capped by a disc
    // with two nostrils -- which is the whole point of the animal.
    bool snout;
    float noseR;     // dog: radius of the nose dome
    float snoutLen;  // pig: how far the snout stands off the face
    float snoutR;    // pig: radius of the tube at the face
    float snoutFlat; // pig: disc height / width
    float snoutDrop; // pig: how far the axis tilts down as it comes forward
    Vec3f noseCol, snoutCol, nostrilCol;
};

Style styleFor(Species sp) {
    Style s{};
    if (sp == Species::Dog) {
        s.earAttachX = 0.88f; s.earAttachY = 0.10f;
        s.earTipX = 0.99f;    s.earTipY = 1.32f;
        s.earHalfW = 0.30f;   s.earRound = 3.2f;
        s.earBulge = 0.15f;   s.earCurl = 0.12f;
        s.earMottle = 0.07f;
        s.earCol = Vec3f(62, 96, 138);
        s.earInnerCol = Vec3f(78, 92, 126);
        s.snout = false;
        s.noseR = 0.24f;
        s.noseCol = Vec3f(30, 28, 28);
    } else { // Pig
        // Ears stand up off the crown and lean outward: the tip's y is
        // negative, i.e. *above* the crown, which is what keeps them clear of
        // the face entirely.
        s.earAttachX = 0.56f; s.earAttachY = 0.12f;
        s.earTipX = 0.90f;    s.earTipY = -0.86f;
        s.earHalfW = 0.39f;   s.earRound = 2.4f;
        s.earBulge = 0.16f;   s.earCurl = 0.10f;
        s.earMottle = 0.05f;
        s.earCol = Vec3f(172, 152, 239);
        s.earInnerCol = Vec3f(122, 98, 202);
        s.snout = true;
        s.snoutLen = 0.34f;
        s.snoutR = 0.46f;
        s.snoutFlat = 0.80f;
        s.snoutDrop = 0.16f;
        s.snoutCol = Vec3f(178, 160, 242);
        s.nostrilCol = Vec3f(96, 74, 150);
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

Mesh buildEar(float side, float wiggle, const Head& h, const Style& st) {
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
    const float hw = h.headHalfW;
    const float cy = h.crownY;
    const Vec3f attach(side * st.earAttachX * hw, cy + st.earAttachY, -0.05f);
    const Vec3f tip(side * st.earTipX * hw, cy + st.earTipY, -0.20f);
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
            const Vec3f col = (pink * (1.f - inF) + inner * inF) *
                              (1.f + st.earMottle * mottle(ti, ai * 7 + (int)side));
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

// The nose: a glossy dome standing off the face at the measured nose tip.
// Wound so its outward face is the front face -- copying the winding of a
// forward-sweeping tube instead leaves it entirely back-facing, and culling
// swallows all but a sliver of rim.
Mesh buildNose(const Head& h, const Style& st) {
    Mesh m;
    m.doubleSided = false; // closed and convex: back faces can be culled
    m.ambient = 0.30f;
    m.spec = 0.55f;
    m.shin = 34.f;

    const Vec3f centre(0.f, h.noseY, h.noseZ);
    const int nSeg = 20, nRing = 7;
    std::vector<std::vector<int>> ring(nRing);
    for (int r = 0; r < nRing; ++r) {
        // 0 at the equator, against the face; 1 at the pole, toward the camera.
        const float lat = 0.5f * kPi * (float)r / (nRing - 1);
        const float cr = std::cos(lat) * st.noseR, cz = std::sin(lat) * st.noseR;
        for (int i = 0; i < nSeg; ++i) {
            const float a = 2.f * kPi * i / nSeg;
            // A little wider than tall, like real nose leather.
            ring[r].push_back(m.add(centre + Vec3f(1.20f * cr * std::cos(a),
                                                   0.88f * cr * std::sin(a),
                                                   -cz),
                                    st.noseCol));
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
// A short elliptical tube standing off the nose, capped by a slightly domed
// disc with two nostrils in it. Both parts share one frame so the nostrils
// cannot drift off the end when the snout moves or resizes.
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
    m.ambient = 0.42f;
    m.spec = 0.26f;
    m.shin = 18.f;
    const SnoutFrame sf = snoutFrame(h, st);
    const int nSeg = 26, nRing = 4;

    std::vector<std::vector<int>> ring(nRing);
    for (int r = 0; r < nRing; ++r) {
        const float t = (float)r / (nRing - 1);
        const Vec3f c = sf.base + sf.axis * (t * sf.len);
        const float rx = sf.rx * (1.f + 0.14f * t); // flares toward the disc
        const float ry = sf.ry * (1.f + 0.12f * t);
        for (int i = 0; i < nSeg; ++i) {
            const float a = 2.f * kPi * i / nSeg;
            ring[r].push_back(m.add(c + sf.u * (rx * std::cos(a)) +
                                        sf.v * (ry * std::sin(a)),
                                    st.snoutCol));
        }
    }
    for (int r = 0; r + 1 < nRing; ++r)
        for (int i = 0; i < nSeg; ++i) {
            const int j = (i + 1) % nSeg;
            m.face(ring[r][i], ring[r][j], ring[r + 1][j]);
            m.face(ring[r][i], ring[r + 1][j], ring[r + 1][i]);
        }

    // The disc closing the tube, domed very slightly toward the camera.
    const Vec3f padCol = st.snoutCol * 1.08f;
    const float frx = sf.rx * 1.16f, fry = sf.ry * 1.14f;
    const int centre = m.add(sf.pad + sf.axis * 0.05f, padCol);
    std::vector<int> rim;
    for (int i = 0; i < nSeg; ++i) {
        const float a = 2.f * kPi * i / nSeg;
        rim.push_back(m.add(sf.pad + sf.u * (frx * std::cos(a)) +
                                sf.v * (fry * std::sin(a)),
                            padCol));
    }
    // Reversed relative to the tube's winding: a fan facing the camera turns
    // the opposite way round from a ring sweeping away from it. Wound the
    // other way the disc is entirely back-facing, culling removes it, and the
    // snout renders as a hollow ring with the tube's inner wall showing.
    for (int i = 0; i < nSeg; ++i)
        m.face(centre, rim[(i + 1) % nSeg], rim[i]);
    m.computeNormals();
    return m;
}

Mesh buildNostrils(const Head& h, const Style& st) {
    Mesh m;
    m.doubleSided = true; // tiny discs; winding must never hide them
    m.ambient = 0.30f;
    m.spec = 0.05f;
    m.shin = 20.f;
    const SnoutFrame sf = snoutFrame(h, st);
    const int nSeg = 16;
    for (float side : {-1.f, 1.f}) {
        // Just in front of the disc so they win the depth test against it.
        const Vec3f c = sf.pad + sf.u * (0.40f * sf.rx * side) +
                        sf.axis * 0.07f;
        const float rx = 0.22f * sf.rx, ry = 0.34f * sf.ry;
        const int centre = m.add(c + sf.axis * 0.02f, st.nostrilCol);
        std::vector<int> rim;
        for (int i = 0; i < nSeg; ++i) {
            const float a = 2.f * kPi * i / nSeg;
            rim.push_back(m.add(c + sf.u * (rx * std::cos(a)) +
                                    sf.v * (ry * std::sin(a)),
                                st.nostrilCol));
        }
        for (int i = 0; i < nSeg; ++i)
            m.face(centre, rim[i], rim[(i + 1) % nSeg]);
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

void render(cv::Mat& frame, const Head& head, Species species) {
    if (frame.empty() || frame.type() != CV_8UC3) return;
    if (head.unit < 12.f) return; // too small to render cleanly

    const Style st = styleFor(species);
    const float wig = 0.05f * std::sin((float)head.phase * 0.11f);
    std::vector<Mesh> meshes;
    meshes.push_back(buildEar(-1.f, wig, head, st));
    meshes.push_back(buildEar(+1.f, wig, head, st));
    if (st.snout) {
        meshes.push_back(buildSnout(head, st));
        meshes.push_back(buildNostrils(head, st));
    } else {
        meshes.push_back(buildNose(head, st));
    }

    // Image-space bounding box of every projected vertex -> the region touched.
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
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (roi.width < 2 || roi.height < 2) return;

    const cv::Point2f org((float)roi.x, (float)roi.y);
    const int LW = roi.width * kSS, LH = roi.height * kSS;
    cv::Mat layer(LH, LW, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat cover(LH, LW, CV_32F, cv::Scalar(0));
    cv::Mat zbuf(LH, LW, CV_32F, cv::Scalar(1e9f));
    for (const Mesh& m : meshes)
        raster(m, head, (float)kSS, org, layer, cover, zbuf);

    cv::Mat colDown, covDown;
    cv::resize(layer, colDown, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_AREA);
    cv::resize(cover, covDown, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_AREA);

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
