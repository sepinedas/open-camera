#include "icons.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <SDL2/SDL2_gfxPrimitives.h>

namespace olc {
namespace {

// Foreground white for most icons; specific ones override the colour.
constexpr Uint8 kFg = 240;

Uint8 mod(Uint8 base, Uint8 alpha) {
    return static_cast<Uint8>((int)base * (int)alpha / 255);
}

// A ring (hollow circle) of the given stroke thickness.
void ring(SDL_Renderer* r, int cx, int cy, int rad, int thick,
          Uint8 cr, Uint8 cg, Uint8 cb, Uint8 a) {
    for (int i = 0; i < thick; ++i)
        aacircleRGBA(r, cx, cy, rad - i, cr, cg, cb, a);
}

void iconShutter(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    ring(r, cx, cy, rad, 3, c, c, c, a);
    filledCircleRGBA(r, cx, cy, rad - 6, c, c, c, a);
}

// Magnifier with a "+" or "-" in the lens.
void iconZoom(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a, bool plus) {
    Uint8 c = mod(kFg, a);
    int lens = (int)(rad * 0.62);
    int lx = cx - (int)(rad * 0.22);
    int ly = cy - (int)(rad * 0.22);
    ring(r, lx, ly, lens, 3, c, c, c, a);
    // handle
    int hx = lx + (int)(lens * 0.72);
    int hy = ly + (int)(lens * 0.72);
    thickLineRGBA(r, hx, hy, cx + rad - 2, cy + rad - 2, 4, c, c, c, a);
    // + / -
    int s = (int)(lens * 0.5);
    thickLineRGBA(r, lx - s, ly, lx + s, ly, 3, c, c, c, a);
    if (plus) thickLineRGBA(r, lx, ly - s, lx, ly + s, 3, c, c, c, a);
}

// Framed landscape (photo) icon: rectangle with a sun and a hill.
void iconGallery(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    int w = (int)(rad * 0.95), h = (int)(rad * 0.72);
    roundedRectangleRGBA(r, cx - w, cy - h, cx + w, cy + h, 4, c, c, c, a);
    filledCircleRGBA(r, cx - w / 2, cy - h / 3, (int)(rad * 0.16), c, c, c, a);
    // hill (triangle) rising to the frame's right edge
    filledTrigonRGBA(r, cx - w + 3, cy + h - 2,
                        cx + (int)(w * 0.2), cy - (int)(h * 0.1),
                        cx + w - 3, cy + h - 2, c, c, c, a);
}

void iconBack(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    int s = (int)(rad * 0.55);
    thickLineRGBA(r, cx + s / 2, cy - s, cx - s, cy, 4, c, c, c, a);
    thickLineRGBA(r, cx - s, cy, cx + s / 2, cy + s, 4, c, c, c, a);
}

// Left/right navigation triangle.
void iconNav(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a, bool left) {
    Uint8 c = mod(kFg, a);
    int s = (int)(rad * 0.6);
    if (left)
        filledTrigonRGBA(r, cx + s, cy - s, cx + s, cy + s, cx - s, cy, c, c, c, a);
    else
        filledTrigonRGBA(r, cx - s, cy - s, cx - s, cy + s, cx + s, cy, c, c, c, a);
}

// Play triangle inside a ring.
void iconPlay(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    ring(r, cx, cy, rad, 3, c, c, c, a);
    int s = (int)(rad * 0.42);
    filledTrigonRGBA(r, cx - s + 2, cy - s, cx - s + 2, cy + s, cx + s + 2, cy, c, c, c, a);
}

void iconTrash(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    int w = (int)(rad * 0.6), top = cy - (int)(rad * 0.5), bot = cy + (int)(rad * 0.6);
    // lid + handle
    thickLineRGBA(r, cx - w - 3, top, cx + w + 3, top, 3, c, c, c, a);
    thickLineRGBA(r, cx - w / 2, top, cx + w / 2, top - (int)(rad * 0.22), 3, c, c, c, a);
    thickLineRGBA(r, cx + w / 2, top, cx + w / 2, top - (int)(rad * 0.22), 3, c, c, c, a);
    // body
    roundedRectangleRGBA(r, cx - w, top + 4, cx + w, bot, 3, c, c, c, a);
    // ribs
    for (int i = -1; i <= 1; ++i)
        thickLineRGBA(r, cx + i * (w / 2), top + 9, cx + i * (w / 2), bot - 4, 2, c, c, c, a);
}

// Smiley face: the live facial-filter (cycle) button.
void iconFilter(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    ring(r, cx, cy, rad, 3, c, c, c, a);
    int eo = (int)(rad * 0.34), ey = cy - (int)(rad * 0.20);
    int er = std::max(2, (int)(rad * 0.12));
    filledCircleRGBA(r, cx - eo, ey, er, c, c, c, a);
    filledCircleRGBA(r, cx + eo, ey, er, c, c, c, a);
    // Upturned smile: an arc approximated by short chords across the lower face.
    int sr = (int)(rad * 0.5);
    int scy = cy - (int)(rad * 0.10);
    int px = cx - sr, py = scy;
    for (int deg = 20; deg <= 160; deg += 20) {
        double t = deg * 3.14159265358979 / 180.0;
        int nx = cx + (int)(sr * -std::cos(t));
        int ny = scy + (int)(sr * std::sin(t) * 0.85);
        thickLineRGBA(r, px, py, nx, ny, 3, c, c, c, a);
        px = nx;
        py = ny;
    }
}

// Little camera silhouette: the "start camera" button on the welcome screen.
void iconCamera(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    int w = (int)(rad * 0.92), h = (int)(rad * 0.6);
    int top = cy - h + (int)(rad * 0.08);
    // viewfinder bump on top
    roundedBoxRGBA(r, cx - (int)(w * 0.38), top - (int)(rad * 0.22),
                   cx - (int)(w * 0.04), top + 2, 2, c, c, c, a);
    // body
    roundedRectangleRGBA(r, cx - w, top, cx + w, cy + h, 4, c, c, c, a);
    // lens
    ring(r, cx, cy + (int)(rad * 0.06), (int)(rad * 0.42), 3, c, c, c, a);
    filledCircleRGBA(r, cx, cy + (int)(rad * 0.06), (int)(rad * 0.15), c, c, c, a);
    // flash dot
    filledCircleRGBA(r, cx + (int)(w * 0.7), top + (int)(rad * 0.12),
                     std::max(2, (int)(rad * 0.08)), c, c, c, a);
}

// Crescent moon: "sleep / blank the screen". Built by scanning the outer disc
// and subtracting an offset disc, so it needs no knowledge of the background.
void iconMoon(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    double R = rad * 0.92;      // outer radius
    double r2 = rad * 0.80;     // carved (bite) radius
    double ox = rad * 0.52;     // carve offset to the upper-right
    double oy = -rad * 0.18;
    for (int dy = -(int)R; dy <= (int)R; ++dy) {
        double outer = R * R - (double)dy * dy;
        if (outer < 0) continue;
        double dxo = std::sqrt(outer);
        int xl = cx - (int)dxo, xr = cx + (int)dxo;
        double yc = dy - oy;
        double inner = r2 * r2 - yc * yc;
        if (inner > 0) {
            double dxc = std::sqrt(inner);
            int cl = cx + (int)ox - (int)dxc, cr = cx + (int)ox + (int)dxc;
            if (cl > xl) hlineRGBA(r, xl, std::min(cl, xr), cy + dy, c, c, c, a);
            if (cr < xr) hlineRGBA(r, std::max(cr, xl), xr, cy + dy, c, c, c, a);
        } else {
            hlineRGBA(r, xl, xr, cy + dy, c, c, c, a);
        }
    }
}

// House: "go back to the welcome (home) screen".
void iconHouse(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    Uint8 c = mod(kFg, a);
    int w = (int)(rad * 0.72);
    int eave = cy - (int)(rad * 0.06);
    int roofY = cy - (int)(rad * 0.62);
    // roof
    filledTrigonRGBA(r, cx - w - 3, eave, cx, roofY, cx + w + 3, eave, c, c, c, a);
    // body outline
    roundedRectangleRGBA(r, cx - (int)(w * 0.78), eave, cx + (int)(w * 0.78),
                         cy + (int)(rad * 0.66), 2, c, c, c, a);
    // door
    int dw = (int)(rad * 0.2);
    boxRGBA(r, cx - dw, cy + (int)(rad * 0.14), cx + dw, cy + (int)(rad * 0.66),
            c, c, c, a);
}

void iconCheck(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    int s = (int)(rad * 0.55);
    thickLineRGBA(r, cx - s, cy, cx - s / 4, cy + s, 5, 90, 210, 110, a);
    thickLineRGBA(r, cx - s / 4, cy + s, cx + s, cy - s, 5, 90, 210, 110, a);
}

void iconCross(SDL_Renderer* r, int cx, int cy, int rad, Uint8 a) {
    int s = (int)(rad * 0.5);
    thickLineRGBA(r, cx - s, cy - s, cx + s, cy + s, 5, 235, 80, 80, a);
    thickLineRGBA(r, cx + s, cy - s, cx - s, cy + s, 5, 235, 80, 80, a);
}

} // namespace

void drawIcon(SDL_Renderer* ren, Action action, int cx, int cy, int r,
              Uint8 alpha) {
    switch (action) {
        case Action::Shutter:     iconShutter(ren, cx, cy, r, alpha); break;
        case Action::ZoomIn:      iconZoom(ren, cx, cy, r, alpha, true); break;
        case Action::ZoomOut:     iconZoom(ren, cx, cy, r, alpha, false); break;
        case Action::OpenGallery: iconGallery(ren, cx, cy, r, alpha); break;
        case Action::Back:        iconBack(ren, cx, cy, r, alpha); break;
        case Action::Prev:        iconNav(ren, cx, cy, r, alpha, true); break;
        case Action::Next:        iconNav(ren, cx, cy, r, alpha, false); break;
        case Action::Play:        iconPlay(ren, cx, cy, r, alpha); break;
        case Action::Delete:      iconTrash(ren, cx, cy, r, alpha); break;
        case Action::ConfirmYes:  iconCheck(ren, cx, cy, r, alpha); break;
        case Action::ConfirmNo:   iconCross(ren, cx, cy, r, alpha); break;
        case Action::CycleFilter: iconFilter(ren, cx, cy, r, alpha); break;
        case Action::StartCamera: iconCamera(ren, cx, cy, r, alpha); break;
        case Action::Sleep:       iconMoon(ren, cx, cy, r, alpha); break;
        case Action::Home:        iconHouse(ren, cx, cy, r, alpha); break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
// Welcome-screen Lego-brick camera
// ---------------------------------------------------------------------------

namespace {

// A classic-Lego colour with helpers to shade it for studs, highlights and
// outlines so each brick reads as a little 3-D block.
struct Brick { Uint8 r, g, b; };

Uint8 shade(Uint8 c, double f) {
    int v = (int)std::lround((double)c * f);
    return (Uint8)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

} // namespace

void drawLegoCamera(SDL_Renderer* ren, int cx, int cy, double u, Uint8 alpha) {
    auto I = [](double v) { return (int)std::lround(v); };
    auto A = [&](Uint8 c) { return mod(c, alpha); };
    const double brickH = 1.18 * u;
    const double studR = 0.30 * u;

    // Classic Lego palette, cycled across the body bricks.
    const Brick pal[] = {
        {201, 26, 9},    // red
        {245, 205, 47},  // yellow
        {0, 85, 191},    // blue
        {35, 120, 65},   // green
        {214, 121, 35},  // orange
        {160, 165, 169}, // light grey
    };
    const int npal = (int)(sizeof(pal) / sizeof(pal[0]));

    // Draw one brick: studs first (their lower halves get painted over by the
    // body, and their upper halves by whatever brick is stacked on top -- so
    // only the topmost row keeps visible studs, exactly like real bricks).
    auto brick = [&](double x, double y, int wU, Brick col) {
        int x0 = I(x), y0 = I(y), x1 = I(x + wU * u), y1 = I(y + brickH);
        int sr = std::max(2, I(studR));
        for (int i = 0; i < wU; ++i) {
            int sx = I(x + (i + 0.5) * u);
            filledCircleRGBA(ren, sx, y0, sr, A(shade(col.r, 1.05)),
                             A(shade(col.g, 1.05)), A(shade(col.b, 1.05)), alpha);
            aacircleRGBA(ren, sx, y0, sr, A(shade(col.r, 0.65)),
                         A(shade(col.g, 0.65)), A(shade(col.b, 0.65)), alpha);
            filledCircleRGBA(ren, sx - std::max(1, I(0.07 * u)),
                             y0 - std::max(1, I(0.05 * u)), std::max(1, I(studR * 0.4)),
                             A(shade(col.r, 1.4)), A(shade(col.g, 1.4)),
                             A(shade(col.b, 1.4)), alpha);
        }
        int rad = std::max(2, I(0.12 * u));
        roundedBoxRGBA(ren, x0, y0, x1, y1, rad, A(col.r), A(col.g), A(col.b), alpha);
        // top highlight + bottom shadow strips for a little depth
        boxRGBA(ren, x0 + 2, y0 + 1, x1 - 2, y0 + std::max(2, I(0.16 * u)),
                A(shade(col.r, 1.22)), A(shade(col.g, 1.22)),
                A(shade(col.b, 1.22)), (Uint8)(alpha * 0.85));
        boxRGBA(ren, x0 + 2, y1 - std::max(2, I(0.18 * u)), x1 - 2, y1 - 1,
                A(shade(col.r, 0.68)), A(shade(col.g, 0.68)),
                A(shade(col.b, 0.68)), (Uint8)(alpha * 0.85));
        roundedRectangleRGBA(ren, x0, y0, x1, y1, rad, A(shade(col.r, 0.55)),
                             A(shade(col.g, 0.55)), A(shade(col.b, 0.55)), alpha);
    };

    // Body: 8 studs wide, 4 brick rows. Rows alternate a straight [2,2,2,2]
    // course with a staggered [1,2,2,2,1] one so the seams look like bricklaying.
    const int cols = 8, rows = 4;
    const double bodyW = cols * u, bodyH = rows * brickH;
    const double bx = cx - bodyW / 2.0;
    const double by = cy - bodyH / 2.0 + 0.55 * u; // leave room for the top bumps

    const std::vector<int> straight = {2, 2, 2, 2};
    const std::vector<int> stagger = {1, 2, 2, 2, 1}; // same 8-stud width, offset seams
    int colorSeed = 0;
    for (int rowk = 0; rowk < rows; ++rowk) { // bottom (0) -> top: upper covers lower
        double ry = by + (rows - 1 - rowk) * brickH;
        const std::vector<int>& widths = (rowk % 2 == 0) ? straight : stagger;
        double bxx = bx;
        for (size_t i = 0; i < widths.size(); ++i) {
            brick(bxx, ry, widths[i], pal[(colorSeed + (int)i) % npal]);
            bxx += widths[i] * u;
        }
        colorSeed += (rowk % 2 == 0) ? 3 : 2; // shift the palette per row
    }

    // --- Top bumps (drawn over the body's top edge) ---
    // Viewfinder: a dark 2-wide brick on the top-left.
    brick(bx + 0.5 * u, by - brickH * 0.86, 2, {33, 33, 38});
    // Shutter button: a red round "1x1" on the top-right.
    {
        int scx = I(bx + 6.5 * u), scy = I(by - 0.34 * u);
        int rr = std::max(3, I(0.5 * u));
        filledCircleRGBA(ren, scx, scy, rr, A(201), A(26), A(9), alpha);
        aacircleRGBA(ren, scx, scy, rr, A(shade(201, 0.6)), A(shade(26, 0.6)),
                     A(shade(9, 0.6)), alpha);
        filledCircleRGBA(ren, scx - std::max(1, I(0.12 * u)),
                         scy - std::max(1, I(0.12 * u)), std::max(1, I(0.16 * u)),
                         A(240), A(120), A(110), alpha);
    }

    // --- Lens: a chunky round assembly on the front, drawn over the body ---
    int lcx = cx, lcy = I(cy + 0.35 * u);
    int lr = I(1.95 * u);
    filledCircleRGBA(ren, lcx, lcy, lr, A(58), A(58), A(64), alpha);       // housing
    aacircleRGBA(ren, lcx, lcy, lr, A(20), A(20), A(24), alpha);
    ring(ren, lcx, lcy, I(1.55 * u), 3, A(28), A(28), A(32), alpha);       // barrel ring
    filledCircleRGBA(ren, lcx, lcy, I(1.2 * u), A(85), A(165), A(225), alpha); // glass
    ring(ren, lcx, lcy, I(1.2 * u), 2, A(30), A(70), A(120), alpha);
    filledCircleRGBA(ren, lcx, lcy, I(0.55 * u), A(38), A(96), A(150), alpha);
    // glass highlight
    filledCircleRGBA(ren, lcx - I(0.45 * u), lcy - I(0.45 * u), I(0.32 * u),
                     A(220), A(240), A(255), (Uint8)(alpha * 0.9));
}

// ---------------------------------------------------------------------------
// Battery gauge
// ---------------------------------------------------------------------------

void drawBattery(SDL_Renderer* ren, int x, int y, int w, int h, int percent,
                 bool charging, Uint8 alpha) {
    percent = std::max(0, std::min(100, percent));

    // Leave room on the right for the terminal nub, so (x, y, w, h) is the
    // whole glyph and callers can lay it out against a screen edge.
    int nub = std::max(2, h / 5);
    int bw = w - nub;
    int bh = h;
    int nubH = std::max(3, h / 2);

    // Charging is its own state: a full green bar while plugged in would read
    // as "done", so keep the level colour for the fill and add the bolt.
    Uint8 fr, fg, fb;
    if (charging)        { fr = 90;  fg = 200; fb = 250; } // blue: taking charge
    else if (percent > 50) { fr = 86;  fg = 196; fb = 112; }
    else if (percent > 20) { fr = 240; fg = 190; fb = 70; }
    else                   { fr = 235; fg = 80;  fb = 70; }

    // Shell: a translucent dark body so the gauge stays legible over a bright
    // preview, then a white outline and the terminal nub.
    Uint8 shell = mod(200, alpha);
    roundedBoxRGBA(ren, x, y, x + bw, y + bh, 3, 12, 14, 20, mod(110, alpha));
    roundedRectangleRGBA(ren, x, y, x + bw, y + bh, 3, shell, shell, shell, alpha);
    boxRGBA(ren, x + bw + 1, y + (bh - nubH) / 2, x + bw + nub,
            y + (bh + nubH) / 2, shell, shell, shell, alpha);

    // Fill, inset by the outline. Always show a sliver so an almost-flat pack
    // still reads as a level rather than an empty shell.
    int pad = 2;
    int inner = bw - 2 * pad;
    int fill = (inner * percent) / 100;
    if (percent > 0) fill = std::max(fill, 2);
    if (fill > 0)
        boxRGBA(ren, x + pad, y + pad, x + pad + fill, y + bh - pad,
                mod(fr, alpha), mod(fg, alpha), mod(fb, alpha), alpha);

    if (!charging) return;

    // Lightning bolt centred on the body: a six-point zig-zag, filled flat so
    // it stays sharp at the small sizes this is rendered at.
    double cx = x + bw / 2.0, cy = y + bh / 2.0;
    double sx = bw / 10.0, sy = bh / 2.6;
    Sint16 bx[] = {(Sint16)std::lround(cx + 0.9 * sx),
                   (Sint16)std::lround(cx - 1.1 * sx),
                   (Sint16)std::lround(cx + 0.1 * sx),
                   (Sint16)std::lround(cx - 0.9 * sx),
                   (Sint16)std::lround(cx + 1.1 * sx),
                   (Sint16)std::lround(cx - 0.1 * sx)};
    Sint16 by[] = {(Sint16)std::lround(cy - sy),
                   (Sint16)std::lround(cy + 0.15 * sy),
                   (Sint16)std::lround(cy + 0.15 * sy),
                   (Sint16)std::lround(cy + sy),
                   (Sint16)std::lround(cy - 0.15 * sy),
                   (Sint16)std::lround(cy - 0.15 * sy)};
    filledPolygonRGBA(ren, bx, by, 6, 255, 255, 255, alpha);
}

} // namespace olc
