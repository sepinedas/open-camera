# open-lego-camera-cpp

A touch-friendly, **icon-only** camera app for the **Raspberry Pi 5**
(or any Linux box with a webcam), written in **C++20**.

- Works with the **Raspberry Pi camera module** (via libcamera / GStreamer) or
  any **USB webcam** (via V4L2) — auto-detected at startup, or selected
  explicitly with `--camera picam` / `--camera webcam`.
- Runs on a **headless Raspberry Pi** with **no desktop, X11 or Wayland** — it
  draws straight to the **HDMI** output through DRM/KMS (SDL2's `kmsdrm`
  driver, selected automatically).
- Opens on a **welcome screen** with a **camera built from Lego bricks** and two
  big controls: **Start Camera** and **Sleep**. Sleep blanks the screen (and, on
  a Raspberry Pi, powers the panel off via `vcgencmd display_power` to save
  energy); a **double-tap** on the screen wakes it. In the camera view a
  **home** button returns to the welcome screen.
- Fullscreen live preview with a **translucent, auto-hiding menu**: a few
  seconds after your last tap the menu fades away; tap anywhere to bring it
  back.
- Menu buttons are **translucent icons, no text**: home, filter, gallery,
  shutter.
- The **live preview fills the whole screen** — it's scaled to the panel's
  aspect ratio (cropping the overflow) so there are no letterbox bars.
- **Pinch-to-zoom** with two fingers (digital, up to 4×); the magnification
  factor (e.g. `2.0x`) shows briefly while zooming.
- A **shutter-flash animation** plays when a photo is taken, and the **gallery
  button shows a thumbnail** of the most recent capture.
- A **battery monitor** for the **Waveshare UPS HAT (D)**: the app reads the
  HAT's INA219 over I²C and shows a **gauge in the top-right corner** — level,
  percentage, a **bolt while charging**, and a pulsing red **LOW BATTERY**
  warning when the cell is nearly flat. Auto-detected; the app runs exactly as
  before when no HAT is fitted.
- Built-in **gallery**: browse captured photos, **play** back any videos
  already on disk, and **delete** items behind an icon-only ✓ / ✗ confirmation.
  The capture **date & time** is shown translucent across the top.
- **WhatsApp-style facial filters** (smiley button): a **Big Smile** that
  stretches your mouth into a wide grin — with your teeth brightening as you
  open it — and a **Crying** face that pulls your mouth and brows into a frown
  and adds animated falling **tears**. These two *reshape the face in place* (its
  own pixels warped), not covered with cartoon graphics — only the tears are
  drawn on top.

![Welcome screen](docs/welcome-screen.png)

![UI mockup](docs/ui-mockup.png)

> The screenshots above are rendered by the offscreen mockup tools
> (`tools/welcome_mockup.cpp` and `tools/mockup.cpp`) using the exact same UI
> code the app runs.

## How it meets the brief

| Requirement | How |
| --- | --- |
| Welcome screen with a Lego-brick camera; Start / Sleep options | `Mode::Welcome` draws `drawLegoCamera` (bricks + lens in `icons.cpp`); Sleep blanks the panel via `vcgencmd display_power` and wakes on a double-tap |
| Runs with a webcam **or** Pi camera | `Camera` auto-detects: libcamera (GStreamer) first, then V4L2 webcam; force one with `--camera picam` / `--camera webcam` |
| Written in C++ | C++20, CMake build |
| Translucent, auto-hiding menu | `Menu` fades the icon row out ~3.5 s after the last tap; any tap wakes it |
| Photos, zoom, gallery, delete | shutter / gallery icons; pinch-to-zoom |
| Icon-only buttons, no text | all icons are drawn as vector shapes (`icons.cpp`, SDL2_gfx) |
| Headless — no X11 / window manager | SDL2 `kmsdrm`/`fbcon` renders directly to HDMI |
| WhatsApp-style facial filters | `FaceFilter` runs MediaPipe's Face Landmarker and warps the real mouth/brow landmarks with `cv::remap`; the crying filter also draws tears (`filters.cpp`) |
| Battery monitor | `Battery` reads the Waveshare UPS HAT (D)'s INA219 over I²C (`battery.cpp`) and `drawBattery` paints the corner gauge (`icons.cpp`) |

## Dependencies

Install the development libraries (names are for Raspberry Pi OS / Debian
Trixie):

```sh
sudo apt install build-essential cmake pkg-config \
                 libsdl2-dev libsdl2-gfx-dev libopencv-dev
```

### MediaPipe (facial filters)

The **facial filters** run **MediaPipe's Face Landmarker** (Tasks Vision C++
API, CPU/TFLite). Google does not publish a C++ SDK, so the Pi build comes from
[**media-pipe-builder**](https://github.com/sepinedas/media-pipe-builder), a
pipeline that compiles MediaPipe for the Pi 5 on 64-bit Raspberry Pi OS
(aarch64) and publishes a `.deb`. Grab the latest release and install it:

```sh
sudo dpkg -i libmediapipe_*_arm64.deb
```

The package **must be built for the same Debian release your Pi runs**. It
links OpenCV by soname, and Raspberry Pi OS has moved: Trixie carries OpenCV
4.10 (`libopencv_core.so.410`), Bookworm 4.6 (`.so.406`). A package built on
the wrong one installs (older releases) or is refused by apt (current ones),
and if it does install the app dies at startup with `error while loading
shared libraries: libopencv_core.so.406`. Installing both OpenCV versions is
not a fix — `cv::Mat` is passed across the MediaPipe boundary, so two OpenCV
ABIs in one process is undefined behaviour. Check yours with:

```sh
. /etc/os-release && echo "$PRETTY_NAME"
```

and build the MediaPipe package with a matching `debian_suite`.

Installing puts MediaPipe in `/opt/mediapipe/<ver>` (with `/opt/mediapipe/current`
pointing at it), registers `libmediapipe_tasks.so` with `ldconfig`, and drops a
`mediapipe.pc` into the system pkg-config path — which is how this project's
CMake finds it. Check with:

```sh
pkg-config --modversion mediapipe
```

For a relocatable tarball (or a MediaPipe built by hand) point CMake at the
install root instead:

```sh
cmake -B build -DMEDIAPIPE_ROOT=/path/to/mediapipe
```

MediaPipe's headers are C++20, which is why the whole project is built as C++20.
They are also developed against Clang; if `g++` chokes on them, configure with
`-DCMAKE_CXX_COMPILER=clang++`.

That package is built with `-mcpu=cortex-a76`, so like this app it is **Pi 5
only** — see [Raspberry Pi 5 tuning](#raspberry-pi-5-tuning) below.

The package also ships the model bundle the filters need,
`face_landmarker.task`, under `share/mediapipe/models/`. The app finds it there
automatically; point it somewhere else with
`--face-model /path/to/face_landmarker.task`. Without a model the app still
runs — the facial filters simply stay inactive.

For the **Pi camera module** you also need the libcamera GStreamer element,
which is what lets OpenCV open the camera without a desktop:

```sh
sudo apt install gstreamer1.0-libcamera gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-base libcamera-tools
```

(A USB webcam needs none of the GStreamer/libcamera packages — it goes
through V4L2 directly.)

### Raspberry Pi 5 tuning

The build and the filter settings target a **Raspberry Pi 5** (BCM2712,
Cortex-A76 @ 2.4 GHz) rather than trying to stay portable across every Pi:

| What | Setting | Why |
| --- | --- | --- |
| Compiler | `-mcpu=cortex-a76 -O3` | Assumes ARMv8.2-A (dot product, FP16, LSE atomics) instead of generic ARMv8-A. **Not portable** — see below. |
| Landmark inference | every frame | MediaPipe's VIDEO mode already re-runs the *detector* only when it loses tracking, which is a better throttle than skipping frames — skipping left the warp visibly lagging the face. |
| Detection image | 640 px wide, in colour | The mesh model crops the face out of this image before resizing to its own input, so a wider image is what lets a small or distant face keep enough detail. Colour beats the grayscale a slower board would settle for. |
| Faces tracked | up to 4 | Per-face mesh inference is affordable here. |
| Mouth/brow warp | separable Gaussian | The inner loop used to call `exp()` once per pixel *per control point* — millions per frame with a large face. The Gaussian factorises into a column term and a row term, so it is now width+height evaluations per point instead of width×height. ~28× faster on the warp itself. |

> **These binaries will not run on a Pi 4, Pi 3 or Zero 2 W.** Those are
> Cortex-A72/A53 and will fault with `SIGILL`. For a portable build, configure
> with `-DOLC_TUNE_PI5=OFF` and use a MediaPipe package built with a matching
> (or generic) `TARGET_MCPU`; the filter settings above stay as they are, so
> expect a much lower frame rate on older boards.

A Pi 5 also has headroom for a higher capture resolution than the 1280×720
default — try `--size 1920x1080`. It is not the default because the camera
pipeline pins the requested size in its caps and has no size fallback: if a
sensor cannot deliver exactly that geometry, the camera fails to open.

> The KiCad design in [`hardware/`](hardware/) is still a **CM4** carrier
> board. It has not been retargeted to a CM5, which would need the power stages
> resized — the CM5 draws substantially more current than a CM4.

### Pi camera notes (including the IMX500 AI camera)

The libcamera GStreamer source is asked for a **processed** pixel format
(`NV12`, then `YUV420`/`RGBx`/`BGRx`/`RGB` as fallbacks). This matters on
sensors like the Sony **IMX500 AI camera**: if the format isn't pinned,
libcamerasrc negotiates the sensor's native Bayer stream
(`2028x1520-SRGGB16/RAW`), which the pipeline can't convert, and it fails to
start. Pinning a processed format avoids that.

#### Preview performance (NV12 → GPU)

The ISP inside the Pi camera already outputs **NV12** (a YUV format), so the
live preview keeps frames in NV12 all the way to the screen and lets the Pi's
**GPU** do the YUV→RGB conversion while it draws — via an `SDL_PIXELFORMAT_NV12`
texture — instead of spending a CPU core on a per-frame `videoconvert`. Pinch
**zoom** is likewise a GPU crop-and-scale (a texture source rect), not a CPU
`resize`. That keeps the preview smooth and low-latency, and leaves the
Cortex-A76 cores free for the filters rather than burning one on colour
conversion.

A frame is only converted to BGR on the CPU when something actually needs the
pixels — taking a photo — so the common "just previewing" case does no colour
conversion or resize on the CPU at all.

The **facial filters** stay on that fast path too. Landmark inference gets a
640 px colour image built by `Camera::nv12ToBGRScaled`, which downscales the
**Y and the interleaved UV planes separately** and converts only the small
result — so the full frame is still never colour-converted, and the cost scales
with the detection size rather than the capture size. Only the **face region**
is then converted to BGR, reshaped, and re-encoded back into the NV12 frame.
The GPU still converts and zooms the whole frame, so filtering costs work
proportional to the face's size on screen rather than a full-frame convert
every frame. (A USB webcam, which delivers BGR, still converts the whole frame
for filters.) Inference runs on **every** frame, with MediaPipe in VIDEO mode
so it only re-runs the face *detector* when it loses tracking.

If the renderer can't sample NV12 textures, or raw NV12 capture won't start, the
app transparently falls back to converting to BGR with libcamera's
`videoconvert` — now spread across **all CPU cores** (`n-threads`) with a
decoupling `queue`, so even the fallback is faster than a single-threaded
convert.

If you have **more than one camera** (e.g. the IMX500 *and* a USB webcam),
`--camera auto` tries the first libcamera camera before falling back to a
webcam. Force the source explicitly with `--camera picam` / `--camera webcam`,
and pick a specific libcamera camera with `--picam-name` — list the ids with:

```sh
rpicam-hello --list-cameras
```

Sanity-check the raw pipeline outside the app with:

```sh
gst-launch-1.0 libcamerasrc ! video/x-raw,format=NV12,width=1280,height=720 \
  ! videoconvert ! autovideosink
```

If that shows a picture, the app will too.

## Build

```sh
cmake -B build -S .
cmake --build build -j
```

The binary is `build/open-lego-camera`.

## Run

```sh
build/open-lego-camera [options]
```

```
  --camera auto|picam|webcam   camera source (default: auto)
  --output-dir DIR             where captures are saved
                               (default: ~/Pictures/open-lego-camera)
  --webcam-index N             force /dev/videoN for a USB webcam
  --size WxH                   requested preview size (default: 1280x720)
  --rotate 0|90|180|270        rotate the whole UI to match a rotated panel
  --camera-rotate 0|90|180|270 rotate only the camera image (preview + captures)
  --touch-rotate 0|90|180|270  extra touch rotation if touch is misaligned
  --touch-flip-x / --touch-flip-y   mirror touch on an axis
  --driver NAME                force SDL video driver (kmsdrm, fbcon, x11)
  --windowed                   run in a window instead of fullscreen
  --face-model PATH            MediaPipe face_landmarker.task for the filters
  --no-battery                 skip the Waveshare UPS HAT (D) battery gauge
  --battery-bus N              I2C bus the UPS HAT is on (default: 1)
  --battery-shutdown           power off at the 3.15 V cut-off (off by default)
  --help                       show this help
```

### Facial filters

Tap the **smiley** button in the camera menu to cycle the live facial filter:
**Big Smile** → **Crying** → **Face Mesh** → **Dog Face** → **Pig Face** → off.
The active
filter's name appears briefly on screen, and the effect is baked into any photo
you then capture.

- **Big Smile** stretches your mouth's corners up and out into a wide grin and
  opens it vertically; the more you open your mouth, the more your teeth are
  brightened, so they "pop". If you are *already* grinning, the pull is eased
  off so the result does not go rubbery.
- **Crying** curls your mouth down into a frown, pinches your inner brows down,
  and streams animated tears down your cheeks.
- **Face Mesh** draws the tracking itself: every landmark as a dot, joined by
  MediaPipe's **own** 2556-edge tessellation, with the feature contours (face
  oval, eyes, brows, irises, lips) picked out over the top in a second colour.
- **Dog Face** paints a dog onto that same mesh — tan coat, dark eye mask,
  pale blaze and muzzle — by filling the tessellation's 852 triangles, then
  adds **3D floppy ears and a glossy nose** over the top. The markings *are*
  the mesh, so they sit on the face and follow every turn, tilt and
  expression; the ears and nose are real shaded geometry, because neither can
  come from a mesh that stops at the face.
- **Pig Face** is the same construction with a different table: pink skin and
  a far finer bristle texture, ears that stand up off the crown instead of
  hanging, and a real snout — a short tube standing off the nose, capped by a
  disc with two nostrils.

The first two filters *warp your actual face* — no cartoon mouth or eyes are pasted on
top; only the crying tears are drawn over the image.

**Following the head's own axes.** The eye line from the MediaPipe mesh gives
the in-plane roll and the scale, and every displacement is applied along those
axes rather than the image's, so a tilted head is reshaped along the face
instead of along the screen.

**The mesh filters' connectivity is MediaPipe's, not ours.** The edge tables
come from `face_landmarks_connections.h` in the Tasks headers — the same ones
its own renderers use — so the wireframe is the canonical topology rather than
a triangulation re-derived here. Both mesh filters draw straight onto the
frame. They used to accumulate into a scratch copy of the region and blend it
back for translucency, which cost a region-sized allocation and two extra
passes over every pixel, every frame, per face — for an effect that at 80–88%
opacity was barely visible. It is the most drawing-heavy filter, and unlike the
warps its cost scales with the
number of landmarks rather than the face's size on screen.

**Dog Face is the same mesh, filled instead of outlined.** MediaPipe stores the
tessellation as edges, but they arrive in consecutive triples that close into a
triangle, so the 852-triangle list is derived from that table rather than
carried as a second one — with a `static_assert` that the structure still
holds, so a future table reshuffle cannot silently produce garbage geometry.

Two things do the polishing. The paint is multiplied by the **subject's own
luminance**, normalised by the mean over the face, so their real modelling —
the shadow under the nose, the line of the lips, the fall-off at the jaw —
survives, and the dog looks painted onto a face rather than pasted over one.
Normalising by the region's mean rather than a constant keeps that working in
a dim room as well as a bright one, and it costs nothing: the paint is opaque,
so the pixel being read is one that is about to be overwritten.

On top of that goes **fur** — two octaves of value noise sampled far finer
across the face than down it, so it stretches into strokes. The markings stay
per-vertex (they are low frequency and interpolate fine); only the fur needs
per-pixel evaluation, which is why the head-frame coordinates are interpolated
alongside the colour. The noise is hashed with integer arithmetic rather than
`sin`, so the per-pixel cost is a handful of integer ops. The ears get a
coarser per-vertex version of the same idea, enough to stop them reading as
moulded plastic beside a furred face.

Each marking is placed in the head's own frame — eye-separation units from the
eye midpoint — so it is defined relative to the eyes and nose and holds through
scale, roll and turn. Every *vertex* is coloured and the triangles interpolate
between them: colouring per triangle instead is simpler but shows all 852
facets, worst exactly where a marking has a hard edge, which turned the nose
into a polygonal star. It is far cheaper than the wireframe — one pass over the
face's pixels, rather than thousands of short anti-aliased lines.

The two animals share one renderer, `face3d`, and differ only by a row in its
style table; the markings likewise differ only by a colour function. Adding a
third is a table entry, not another renderer.

**The ears and muzzle are oriented by a basis read off the mesh in 3D.**
MediaPipe gives every landmark a depth, so the head's axes come straight from
it: the outer eye corners span its width, forehead-to-chin its height, and
their cross product the direction it faces. No yaw inferred from how the nose
divides the face, and no foreshortening correction — measuring the ear
attachment and nose position *in that frame* is exact, because nothing in it
is foreshortened. `dog3d` then renders them with a z-buffer, Gouraud shading
and 2×2 supersampling, scaled in eye separations so they track distance.

The dirty region has to allow for all of that. The mesh filters only touch
the landmarks, so a 3 px margin suffices — but the animals hang geometry well
outside the face box: an ear reaches ~1.3x the head half-width, and a pig's
stand a third of a face-height above the crown. They get their own, far more
generous margin. Too small a margin here does not merely lose the saving, it
slices the ears off against the edge of the re-encoded region.

> Three things bite anyone touching this geometry. The scale unit is the
> distance between eye **centres**; using the outer corners instead — an easy
> substitution — silently enlarges the entire rig by about 1.45×. And the nose
> dome must be wound so its outward face is the front face: copying the
> winding of a forward-sweeping tube leaves it wholly back-facing, and culling
> eats all but a sliver of rim. The same trap catches the pig's snout disc,
> which is a *fan* and therefore winds the opposite way round from the tube it
> caps -- get it wrong and the snout renders as a hollow ring with the tube's
> inner wall showing through.

### Rotating the display

`--rotate 90|180|270` rotates the **entire UI** — the camera preview *and* the
icon buttons — clockwise to match a panel mounted in a different orientation
(the UI is drawn to an offscreen canvas and blitted rotated, and taps are
un-rotated to match). Use this when the picture and buttons appear sideways:

```sh
build/open-lego-camera --rotate 90
```

`--rotate` also rotates touch input to match, so if the touch panel is aligned
with the display you usually need nothing else. `--touch-rotate` /
`--touch-flip-x` / `--touch-flip-y` are a *separate* correction for when the
touch controller is mounted rotated/mirrored **relative to the panel** (common
on the HyperPixel) — reach for them only if taps are still off after `--rotate`.

`--camera-rotate 90|180|270` rotates **only the camera image** — the live
preview *and* the photos you capture — while leaving the icon buttons and
the rest of the UI exactly where they are. Reach for it when the panel is
mounted the right way up but the camera module itself sits sideways, so the
picture comes in rotated but the controls don't need to move:

```sh
build/open-lego-camera --camera-rotate 90
```

The image spins on the GPU for the preview (no extra CPU cost) and is baked into
captures so a saved photo matches what you saw. It stacks with `--rotate`, which
still turns the whole UI on top.

- Photos are saved as `IMG_YYYYMMDD_HHMMSS.jpg`. The gallery can still play
  back any existing `.mp4` videos in the output directory.
- The app opens on the **welcome screen**; tap **Start Camera** to begin or
  **Sleep** to blank the screen (**double-tap** to wake). The **home** icon in
  the camera menu returns here.
- **Tap the screen** to wake the menu after it has faded.
- **Esc** or **Q** steps back one screen: camera → welcome, gallery → camera,
  and quits from the welcome screen. Any key wakes the screen from sleep.
- `--windowed` is handy when developing on a desktop (the app then uses the
  desktop's SDL driver automatically).

## Battery monitor (Waveshare UPS HAT (D))

With a [Waveshare UPS HAT (D)](https://www.waveshare.com/wiki/UPS_HAT_(D)) on
the Pi's 40-pin header, the app shows a **battery gauge in the top-right
corner** of the welcome, camera and gallery screens:

| Gauge | Meaning |
| --- | --- |
| green / amber / red fill + `NN%` | charge level (>50% / >20% / below) |
| blue fill with a **lightning bolt** | the cell is taking charge |
| pulsing red + `LOW BATTERY` | at or below 15%, off charge |

Nothing needs to be passed on the command line: at startup the app looks for the
HAT's **INA219** at `0x43` on `/dev/i2c-1` and simply runs without a gauge if
nothing answers. Two things have to be true on the Pi first:

```sh
# 1. I2C enabled -- add to /boot/firmware/config.txt (or use raspi-config), then reboot
dtparam=i2c_arm=on

# 2. your user allowed to use it (log out and back in afterwards)
sudo usermod -aG i2c "$USER"

# check the HAT is answering: 43 (gauge) and 2d (power-path MCU) should appear
i2cdetect -y 1
```

The register map, calibration (0.01 Ω shunt, 16 V / 5 A profile) and the
state-of-charge curve all follow Waveshare's own reference driver for this
board, so the readings match its `INA219.py` demo. Note that the percentage is
**estimated from the cell's terminal voltage** (3.0 V empty → 4.2 V full), not
counted in coulombs, so it sags under a heavy load and recovers when the load
drops; the app smooths it so the number doesn't flicker. The `(B)` and `(C)`
HATs use different shunts and calibration values and are *not* interchangeable
with this code.

Useful flags:

- `--battery-bus N` — the HAT is on a bus other than `/dev/i2c-1`.
- `--no-battery` — skip the probe entirely.
- `--battery-shutdown` — **opt-in**: when the cell stays below the HAT's
  **3.15 V cut-off** for a minute while off charge, show `BATTERY EMPTY`, ask
  the HAT's MCU to power the Pi back up by itself once the cell recovers
  (register `0x01` ← `0x55` at `0x2d`), then halt. Without this flag the app
  only ever *reports* the level. The shutdown runs `sudo -n poweroff`, so it
  needs passwordless sudo (the default for the `pi` user) or root.

> The KiCad [CM4 carrier board](hardware/README.md) in this repo takes a
> different route — an on-board **MAX17048** fuel gauge — which this code does
> not read. The UPS HAT above is the off-the-shelf option for a normal
> 40-pin Pi.

## Headless HDMI (no desktop)

The app does **not** need a desktop, X11 or Wayland. On a Pi booted to the
plain text console (Raspberry Pi OS Lite, or `raspi-config` → *System Options*
→ *Boot / Auto Login* → *Console*) it draws directly to the HDMI screen via
DRM/KMS.

1. Give your user access to the GPU/DRM and input devices once, then re-login:

   ```sh
   sudo usermod -aG video,render,input "$USER"
   ```

2. Run it **from a console on the Pi itself** (a keyboard/screen on the Pi, or
   the active TTY) — not over SSH. SDL needs the active HDMI console to take
   over the framebuffer:

   ```sh
   build/open-lego-camera
   ```

The driver is auto-selected: a desktop driver when `DISPLAY`/`WAYLAND_DISPLAY`
is set, otherwise `kmsdrm` then `fbcon`. Force one with `--driver kmsdrm` (or
set `SDL_VIDEODRIVER`) if the guess is wrong.

> Running over SSH with no HDMI console attached fails with "could not open a
> display" — that is expected; launch it on the Pi's own console.

### Autostart on boot (optional)

On a headless Pi the app must own the **active HDMI console** to grab the
DRM/KMS framebuffer, so the simplest reliable autostart is console auto-login
plus a launch from the login shell.

1. `raspi-config` → *System Options* → *Boot / Auto Login* → *Console
   Autologin*.
2. Append to `~/.bash_profile`:

   ```sh
   # start the camera on the main HDMI console only
   if [ "$(tty)" = "/dev/tty1" ]; then
     exec "$HOME/open-lego-camera-cpp/build/open-lego-camera"
   fi
   ```

`exec` replaces the login shell with the app; `Esc`/`Q` (or a crash) drops you
back to a login prompt.

## Debugging HDMI / no display

Black screen, or "could not open a display"? The output chain is
**SDL2 → `kmsdrm` → DRM/KMS → HDMI connector**; work down it.

**First, read the app's own startup log.** It now prints what it tried:

```
display: driver 'kmsdrm' up; 1 output(s) detected
  [0] HDMI-A-1 1920x1080
display: kmsdrm 1920x1080
```

- `0 output(s) detected` → the kernel sees **no connected HDMI** with a mode
  (cable/EDID/hotplug). Jump to step 3.
- `CreateWindow failed: ...` → a driver/permission/DRM-master problem. Steps 1–2.
- No `display:` lines at all, or it exits immediately → not a display issue;
  check the camera line above it.

**1. Are you on the console, not SSH?** `kmsdrm` must be **DRM master**, which
means running on the machine's active screen, not a bare SSH shell where the
`getty` on tty1 holds it. Run it on the Pi's own keyboard/console, from tty1, or
as a systemd service on the seat. Quick test from SSH: switch the active VT with
`sudo chvt 1` first, or just `sudo build/open-lego-camera`.

**2. Permissions / contention.**
```sh
groups                      # need: video, render, input
sudo usermod -aG video,render,input "$USER"   # then re-login
```
If a desktop (X/Wayland) or another instance is running it will own DRM master
and `CreateWindow` fails — boot to console (`raspi-config` → *System Options* →
*Boot / Auto Login* → *Console*) or stop the desktop.

**3. Does the kernel even see the HDMI connector?**
```sh
ls /dev/dri/                                   # expect card0/card1 + renderD128
for s in /sys/class/drm/*/status; do echo "$s = $(cat "$s")"; done
```
Look for a line like `card1-HDMI-A-1/status = connected`. If it says
`disconnected` while a monitor is plugged in, it's a hotplug/EDID issue — in
`/boot/firmware/config.txt`:
```ini
hdmi_force_hotplug=1
# if still blank, pin a known-good mode (1080p60):
hdmi_group=1
hdmi_mode=16
```
(These `hdmi_*` keys apply to the legacy path; on KMS you can instead force a
mode with a kernel arg in `cmdline.txt`, e.g. `video=HDMI-A-1:1920x1080@60`.)

**4. Is KMS actually enabled?** `kmsdrm` needs the full KMS driver:
```ini
# /boot/firmware/config.txt
dtoverlay=vc4-kms-v3d
```
Check it loaded: `dmesg | grep -i "drm\|vc4"`.

**5. Prove the pipe independent of this app.** Draw a KMS test pattern straight
to the connector (no SDL, no app):
```sh
sudo apt install libdrm-tests   # provides modetest
modetest -M vc4 -c              # list connectors + modes
sudo modetest -M vc4 -s <connector_id>:<mode>   # e.g. -s 32:1920x1080
```
If `modetest` shows nothing either, the problem is entirely system-level
(config.txt / cable / KMS), not the app. If `modetest` works but the app is
black, tell me and we'll dig into SDL.

**6. HDMI + HyperPixel together.** With the HyperPixel DPI overlay enabled there
are two connectors; `kmsdrm` renders to the **first connected** one, which may
be the DPI panel, leaving HDMI dark (or vice-versa). To test HDMI alone,
comment out the `dtoverlay=vc4-kms-dpi-hyperpixel4` line and reboot. To pick one
deliberately, force it in `cmdline.txt` with `video=HDMI-A-1:1920x1080@60`
(and/or disable the other connector).

## Pimoroni HyperPixel 4.0 (DPI touchscreen)

The HyperPixel 4.0" rectangular is an **800×480 DPI panel** (parallel RGB over
the GPIO header) with an **I²C capacitive touch** controller — not HDMI and not
DSI. On a current Raspberry Pi OS (`vc4-kms-v3d`) it comes up as a
normal **DRM/KMS** output, so this app drives it through the same `kmsdrm` path
— you just need to enable the panel and align the touch.

### 1. Enable the panel

Add the HyperPixel4 KMS overlay to `/boot/firmware/config.txt` (make sure the
KMS driver is active and the panel isn't fighting HDMI for the console):

```ini
# Full KMS (usually already present)
dtoverlay=vc4-kms-v3d

# HyperPixel 4.0 rectangular. (Square panel: vc4-kms-dpi-hyperpixel4sq)
dtoverlay=vc4-kms-dpi-hyperpixel4
```

Then reboot and confirm the overlay is actually installed and the panel is a
DRM connector:

```sh
ls /boot/firmware/overlays | grep -i hyperpixel   # overlay present?
```

If your OS image doesn't ship the overlay, install Pimoroni's driver first,
which adds the overlay and the touch setup, then reboot:

```sh
git clone https://github.com/pimoroni/hyperpixel4
cd hyperpixel4 && sudo ./install.sh
```

> **HDMI + HyperPixel together:** SDL's `kmsdrm` renders to the first connected
> connector, which may be HDMI. For a HyperPixel-only setup, leave HDMI
> unplugged (or force the connector with a `video=` kernel arg). The console
> should appear on the HyperPixel once the overlay is active.

### 2. Run it

It's a KMS output, so nothing special is needed — the driver auto-selects
`kmsdrm`:

```sh
build/open-lego-camera            # add --driver kmsdrm to force it
```

The UI is fullscreen and adapts to the panel's 800×480 automatically. If the
preview and buttons come up **sideways** for how the panel is mounted, rotate
the whole UI with `--rotate` (see [Rotating the display](#rotating-the-display)):

```sh
build/open-lego-camera --rotate 90    # try 90/180/270
```

### 3. Align the touch

The HyperPixel's touch controller is commonly **rotated/mirrored** relative to
the panel, so a tap may land in the wrong place. Correct it entirely in the app
— no X11/libinput calibration needed — with:

```sh
build/open-lego-camera --touch-rotate 90                 # try 0/90/180/270
build/open-lego-camera --touch-rotate 90 --touch-flip-x  # add flips if needed
```

Find the right combination by tapping the shutter (bottom row) and watching
where the press registers: pick the `--touch-rotate` that makes vertical taps
track vertically, then add `--touch-flip-x`/`--touch-flip-y` if an axis is
mirrored. If you rotate the **display** via the overlay (a `rotate=` parameter),
match `--touch-rotate` to it. Once found, bake the flags into your autostart
line.

## Menu icons

| Icon | Action |
| --- | --- |
| camera (welcome) | start the live camera |
| crescent moon (welcome) | sleep — blank the screen; double-tap to wake |
| house (camera) | back to the welcome screen |
| last-shot thumbnail (framed-landscape icon until the first capture) | open the gallery |
| ring with dot | take a photo (plays a shutter flash) |
| chevron (gallery) | back to the camera |
| ◀ / ▶ triangles (gallery) | previous / next item |
| triangle-in-ring (gallery) | play the selected video |
| trash can (gallery) | delete the shown item (asks ✓ / ✗) |
| ✓ green / ✗ red | confirm / cancel a delete |

**Zoom** is not a button: **pinch with two fingers** on the preview to zoom
(digital, up to 4×). The current factor (`1.0x`–`4.0x`) appears briefly at the
top while you pinch.

## Design notes

- **Modules** (`src/`): `camera` (dual backend + digital zoom), `gallery`
  (list/navigate/delete), `battery` (UPS HAT (D) INA219 over I²C), `icons`
  (procedural vector icons), `ui` (auto-hide menu, layout, hit-testing), `app`
  (SDL display, event loop, per-mode rendering), `config` (CLI).
- **Zoom** is a uniform centre-crop-and-rescale applied to both preview and
  captures, so behaviour is identical on the Pi camera and a webcam.
- **Rendering**: each frame is uploaded to a streaming SDL texture and scaled to
  **cover** the screen (cropping the overflow so there are no letterbox bars);
  the translucent menu is composited on top with alpha blending.
- Video **playback** decodes frames with OpenCV — no external player needed;
  tap anywhere to stop.

## Preview the UI without a Pi

`tools/mockup.cpp` renders every menu state to `build/ui-mockup.png` using the
real UI code (handy for tweaking icons on a desktop):

```sh
g++ -std=c++17 tools/mockup.cpp src/ui.cpp src/icons.cpp -o build/mockup \
    $(pkg-config --cflags --libs sdl2 SDL2_gfx opencv4)
./build/mockup
```

## License

MIT — see [LICENSE](LICENSE).
