# open-lego-camera-cpp

A touch-friendly, **icon-only** camera app for the **Raspberry Pi 5**
(or any Linux box with a webcam), written in **C++20**.

- Works with the **Raspberry Pi camera module** (via libcamera / GStreamer) or
  any **USB webcam** (via V4L2) — auto-detected at startup, or selected
  explicitly with `--camera picam` / `--camera webcam`.
- **Two cameras at once**: with a USB webcam plugged in alongside the Pi camera
  (or with two webcams), a **switch-camera button** appears in the camera menu
  and flips the live preview between them. The button only shows up when there
  is a second camera to switch to.
- Runs on a **headless Raspberry Pi** with **no desktop, X11 or Wayland** — it
  draws straight to the **HDMI** output through DRM/KMS (SDL2's `kmsdrm`
  driver, selected automatically).
- Opens on a **welcome screen** with a **camera built from Lego bricks** and two
  big controls: **Start Camera** and **Sleep**. Sleep switches the LCD's
  **backlight** and the **display pipeline** off and **closes the camera**, so
  the Pi idles while it waits (see [Sleep and power](#sleep-and-power)); a
  **double-tap** on the screen wakes it. In the camera view a
  **home** button returns to the welcome screen.
- Fullscreen live preview with a **translucent, auto-hiding menu**: a few
  seconds after your last tap the menu fades away; tap anywhere to bring it
  back.
- Menu buttons are **translucent icons, no text**: home, filter, switch camera
  (only with a second camera attached), gallery, shutter.
- The **live preview fills the whole screen** — it's scaled to the panel's
  aspect ratio (cropping the overflow) so there are no letterbox bars.
- **Pinch-to-zoom** with two fingers (digital, up to 4×); the magnification
  factor (e.g. `2.0x`) shows briefly while zooming.
- A **shutter-flash animation** plays when a photo is taken, and the **gallery
  button shows a thumbnail** of the most recent capture.
- A **battery monitor** for the **Waveshare UPS HAT (B)** and **(D)**: the app
  reads the HAT's INA219 over I²C and shows a **gauge in the top-right corner** — level,
  percentage, a **bolt while charging**, and a pulsing red **LOW BATTERY**
  warning when the cell is nearly flat. Auto-detected; the app runs exactly as
  before when no HAT is fitted.
- Built-in **gallery**: browse captured photos, **play** back any videos
  already on disk, and **delete** items behind an icon-only ✓ / ✗ confirmation.
  It reads like a timeline: it opens on the newest capture, and **◀ goes back
  to older ones** (▶ comes forward again). **Pinch a photo to zoom in** (up
  to 6×, about your fingers) and **drag with one finger to pan** around it.
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
| Welcome screen with a Lego-brick camera; Start / Sleep options | `Mode::Welcome` draws `drawLegoCamera` (bricks + lens in `icons.cpp`); Sleep turns the backlight and display pipeline off and closes the camera (`DisplayPower`, `App::enterSleep`); wakes on a double-tap |
| Runs with a webcam **or** Pi camera | `Camera` auto-detects: libcamera (GStreamer) first, then V4L2 webcam; force one with `--camera picam` / `--camera webcam` |
| Switch between two cameras while running | `Camera::sources()` enumerates the V4L2 nodes up front; the switch button reopens the preview on the next one (`App::switchCamera`) |
| Written in C++ | C++20, CMake build |
| Translucent, auto-hiding menu | `Menu` fades the icon row out ~3.5 s after the last tap; any tap wakes it |
| Photos, zoom, gallery, delete | shutter / gallery icons; pinch-to-zoom |
| Icon-only buttons, no text | all icons are drawn as vector shapes (`icons.cpp`, SDL2_gfx) |
| Headless — no X11 / window manager | SDL2 `kmsdrm`/`fbcon` renders directly to HDMI |
| WhatsApp-style facial filters | `FaceFilter` runs MediaPipe's Face Landmarker and warps the real mouth/brow landmarks with `cv::remap`; the crying filter also draws tears (`filters.cpp`) |
| Battery monitor | `Battery` reads the INA219 on a Waveshare UPS HAT (B) or (D) over I²C, with the per-model constants in one board table (`battery.cpp`); `drawBattery` paints the corner gauge (`icons.cpp`) |

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

### More than one camera

With **more than one camera** attached (e.g. the IMX500 *and* a USB webcam, or
two webcams), a **switch-camera button** — a camera inside two circling arrows —
appears in the camera menu, between the filter and gallery buttons. Tapping it
moves the live preview to the next camera and shows its name briefly on screen;
tapping again cycles back round. The zoom you had set is kept across the switch,
and captures come from whichever camera is live.

The list is built at startup: `--camera auto` puts the Pi camera first and then
every `/dev/video*` node that reports itself as a real video-capture device
(`VIDIOC_QUERYCAP`), which skips the metadata nodes UVC webcams expose and the
Pi's own ISP/CSI nodes. A camera that won't open is dropped from the list, so
the button never offers a device that isn't there — and it isn't drawn at all
when only one camera works, leaving the usual four buttons.

`--camera picam` / `--camera webcam` narrow the list to one kind (so they also
turn the switch button off unless you have two webcams), `--webcam-index N`
pins a single `/dev/videoN`, and `--picam-name` picks a specific libcamera
camera — list the ids with:

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
  --mirror none|picam|webcam|all  mirror these cameras like a selfie camera
                               (preview + captures; default: picam)
  --touch-rotate 0|90|180|270  extra touch rotation if touch is misaligned
  --touch-flip-x / --touch-flip-y   mirror touch on an axis
  --driver NAME                force SDL video driver (kmsdrm, fbcon, x11)
  --windowed                   run in a window instead of fullscreen
  --face-model PATH            MediaPipe face_landmarker.task for the filters
  --no-battery                 skip the Waveshare UPS HAT battery gauge
  --battery-hat auto|b|d       which UPS HAT to expect (default: auto-probe)
  --battery-bus N              I2C bus the UPS HAT is on (default: 1)
  --battery-range EMPTY:FULL   pack volts at 0% and 100% (calibration)
  --battery-shutdown           power off at the pack cut-off (off by default)
  --help                       show this help
```

### Facial filters

Tap the **smiley** button in the camera menu to cycle the live facial filter:
**Big Smile** → **Crying** → **Face Mesh** → **Dog Face** → **Pig Face** →
**Grinch** → **Squirrel** → **Elephant** → **Dinosaur** → **Dragon** →
**Shark** → off.
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
  a far finer bristle texture, broad blunt ears that stand up off the crown
  instead of hanging, and a real snout — a short tube standing off the nose,
  capped by a domed disc with two nostrils **cut into** it.
- **Grinch** is the third of that family: vivid green skin that grows shaggier
  toward the crown, yellow-green eye pads, and a pair of heavy brows angled
  down toward the nose — the scowl is drawn with *rotated* ellipses, since an
  axis-aligned one cannot slope. Over it sit 3D pointed elf ears, anchored at
  temple height where the head is genuinely widest, and a small upturned snub
  nose lit as skin rather than as a dog's wet leather.

  **Ears are shells, not sheets.** All three species' ears are built as a
  closed shell: a concha hollowed into the front, a rim standing proud of it,
  a back, and a real front-to-back thickness you can see at the silhouette,
  the whole lobe rolled about its own axis so it faces outward. A single
  convex sheet — which is what these were — reads as a horn however it is
  shaded or shaped. The bowl depth is signed, so the one builder covers both
  an ear turned toward you (pig, grinch) and the *back* of one that hangs
  (dog). Nostrils are likewise carved into the muzzle along its own surface
  normal rather than pasted in front of it.

  (The character is Dr. Seuss's; the geometry and palette here are ours.)
- **Squirrel**, **Elephant**, **Dinosaur**, **Dragon** and **Shark** are not paint at all: each
  replaces the whole head with a 3D model. Nothing is drawn on the mesh — the mesh *drives* the
  model instead. The same measured basis orients it, the head's own crown,
  chin and temples size it to the face it is worn by, and its expression works
  the jaw, the eyelids, the tongue and the cheeks.

  All five are the same construction: a skull and a jaw that hinges against
  it,
  each swept along the head's own longitudinal axis as a single ring grid —
  an outer arc, then a return along the mouth line — so the inside of the
  mouth closes itself and one triangle pattern winds the whole thing. Behind
  the hinge the return bulges into the other half of the ellipse, which is
  what makes the head solid there rather than two shells with a slot between
  them. What differs is a table of curves and colours (`HeadShape`), plus the
  trim hung off it.

  * The **shark** is a long smooth cone: a rostrum, two rows of teeth, gill
    slits, a dorsal fin, and a jaw that drops nearly 50 degrees, which is
    about what a real one manages.
  * The **squirrel** is a round cranium, widest at the cheeks, with a short
    blunt muzzle stepping out of its lower front. That outline is *not* any
    exponent, which is how the first attempt came out looking like a shark in
    a brown coat: a single monotonic taper can only make a cone. So the
    squirrel's silhouette is given as control points instead, sampled along
    the head, and which description a species uses is a function pointer in
    its shape. Over it go big round ears
    that answer the brows, large forward eyes, a nose on the point of the
    muzzle, **buck teeth** hanging just under it — placed in front of the
    muzzle's face, or the closed jaw simply swallows them — and **cheek
    pouches** that fill out on `cheekPuff`, which is the one blendshape
    nothing else here had a use for.

    Its eyes are placed by coordinate rather than by angle round the
    cross-section, the way the shark's are. That works for a shark because its
    sections are not far off round; on a cranium two head-lengths tall, the
    angle that puts an eye at the right height puts it out on the silhouette.
  * The **elephant** is barely about its silhouette at all: what says elephant
    is the ears and the trunk, and those are trim. The head under them is a
    broad domed skull — wider than it is high, which is the opposite of what
    the first attempt gave it — with small eyes set low and wide, because big
    ones on a head that shape make it a cartoon mouse.

    The trunk and the two tusks are one builder: a tube that tapers along a
    *curving* path, where the path is integrated step by step as its direction
    turns rather than written down as a polyline, so it bends smoothly instead
    of having corners in it. Past a quarter turn the tip is rising, which is
    what lets `jawOpen` raise and curl the trunk — an elephant about to
    trumpet — and `mouthPucker` curl just its tip.

    Its ears are nearly the size of its head, and getting that scale right is
    most of what makes the filter read. They needed hand-scaling: `Style`'s
    ear offsets are absolute eye separations rather than fractions of the
    head, so at the values that suit a dog they came out as two small flaps up
    by the crown.

  * The **dinosaur** is a cartoon T. rex: a deep skull, wide at the jaw
    muscles, stepping in to a long boxy snout about half as wide that hangs
    well below the chin. That step is what makes the snout read head-on as a
    separate form under the cheeks rather than the whole head being one
    green egg. It uses the squirrel's control-point outline, but with the
    front of the snout capped by a dome (`domedSection`): closed only in
    girth, the loft ends in a vertical blade, which on a snout this broad
    showed as a crease down the middle of the face.

    The jaw is hinged far back, so the gape runs most of the length of the
    head and shows the shark's teeth, fewer and bigger. A hinge that far back
    exposed something the others got away with: behind the hinge the jaw is a
    whole cross-section of the head, so swinging it rigidly lifted a
    skull-sized cap over the eyes. Every head but the shark (whose hinge is
    right at the back) now fades the swing, and the sideways slide, in
    across the hinge, so only what is in front of it moves. The squirrel and
    elephant needed it too once they were sized from the face at rest, which
    is longer in the jaw. Where the jaw stays put it is tucked a hair inside
    the skull, or the two coincide and z-fight into specks on top of the head.

    Over it go amber eyes with a vertical slit pupil, a bony horn over each
    eye that stands up on raised brows and splays out on a scowl or a sad
    face, a crest of spikes down the middle of the skull — from the front,
    the silhouette that says dinosaur — nostrils on top of the snout, and
    dark bands across the back that fade out before the pale throat.

  * The **dragon** is built on the dinosaur — the same loft, toothed jaw and
    hinge — with a leaner outline: crimson scales over a gold throat, a snout
    that slopes away from the brow and narrows to a muzzle, and fangs rather
    than rows of teeth. Over it go gold slit-pupilled eyes, horns rising off
    the skull and curling back, pointed fins swept out from the sides (raised
    and laid back by the brows, like the animals' ears), a crest of spikes,
    spikes at the corners of the jaw, and nostrils on the snout.

    **Open your mouth and it breathes fire.** The fire fades in as the mouth
    opens past about a third (talking does not set it off) and is at full
    blast near fully open. It is particles, not geometry: ~170 flames stream
    from between the jaws, widening, swirling and rising as they cool. Each
    one deposits *heat*, and the summed heat is coloured through a fire
    palette — nothing below a threshold, then deep red, orange, yellow, and
    white only at the hottest spots. That threshold is what gives the flames
    edges; colouring each flame and adding them up was tried first and gave a
    soft ball in a pink haze. Each flame is drawn as a teardrop that licks
    upward, so the edges come out as tongues rather than a cauliflower.
    The jet points mostly *at* the camera: head-on that reads as a fireball
    swelling out of the jaws toward you, where a jet aimed downward just
    leaves the bottom of the frame.

    The heat is splatted at a quarter of the resolution, and the composite
    (smoke, then flame, then a screened glow, which together are one
    multiply-add per channel) is worked out per coarse node and only
    interpolated per pixel. Counted in `bounds()`, so the preview's dirty
    region always covers the flames.

  Two things about all of them are deliberately not anatomical, because the camera
  only ever sees them from the front:

  * **The muzzle points down as well as forward.** The projection is
    orthographic, so a muzzle aimed at the lens has no length on screen at
    all. It also has to reach past the chin: anything that tapers forward
    *inside* the head hides behind the largest cross-section, which is the one
    that has to cover the head in the first place.
  * **The countershading is keyed to height, not to the cross-section.**
    Head-on, almost the entire visible surface is the animal's dorsal third —
    the pale belly faces the floor and shows as a hairline at the silhouette.
    Shaded honestly the shark comes out uniformly grey and the squirrel
    uniformly brown.

  The squirrel's eyes are a third such compromise: a real one's are far round
  the side of its head, and brought that far round only one of them reads as
  an eye from the front, so they are pulled toward the midline.

**Paying for a head-sized model.** The shark first ran at 61 ms a frame where
the other 3D filters ran at 8–12, which is a different thing entirely on a Pi.
Profiling put 48 of those 61 ms in the rasteriser, and four changes took it to
23 — the first three bit-for-bit identical in output, verified by diffing
renders:

| | |
|---|---|
| `std::pow` per supersample for the specular | exponent is a small constant per mesh, so exponentiate by squaring — **12 ms** |
| lofts emitted back-to-front, the worst order for a z-buffer | sort the shark's shells near-first, so hidden pixels lose the depth test before they are shaded rather than after — **7 ms** |
| three float buffers reallocated every frame | keep and reuse them; at 2x supersampling they are tens of megabytes — **4 ms** |
| supersampling costs its square, over the whole region the model covers | spend a fixed sample budget (`kMaxSamples`) instead of a fixed factor — **16 ms** |

The last one is the only one that changes the picture. Rather than choosing
between 1x and 2x — which puts a hump in the cost curve just under the
threshold and makes edges visibly pop as someone leans in — the factor eases
down continuously once the model outgrows the budget, so a frame costs about
the same whatever is on screen. At the budget set here the ears, muzzles and
grinch are at the full factor for any normal framing and render identically to
before; only the shark reaches it.

`dirtyRegion()` also asks `face3d::bounds()` for the model's exact projected
extent now, rather than padding the face box by margins measured from a sweep
of poses. That was sizing every frame's colour conversion for the worst case
— and left any pose outside the sweep free to be clipped.

### What the filters read off your face

**The 3D models follow the face mesh point by point.** Every one of the 468
landmarks has moved from where it sits on your face at rest by exactly what
your face is doing there. So the models aren't limited to a dozen named
scores: a lopsided smile, a sneer, a pucker, a dropped lower lip or a jaw
worked sideways all move the model the way they move you. It works in three
steps:

1. **Rest.** `detect()` measures each face in its own head frame and keeps a
   per-face mesh *at rest*. It's seeded from the first frame and learnt only
   while the blendshape scores say the face is relaxed, so a held smile isn't
   learnt as neutral. The expression is `live − rest`, landmark by landmark.
   The head frame's vertical axis runs forehead → under the nose, not
   forehead → chin as it used to. The chin drops when the mouth opens, and
   that pitched the whole frame by ~10°, which would now read as motion
   everywhere on the face. At rest the new frame matches the old one to
   0.02°, so every model still sits where it was tuned. The models are also
   *sized* from the rest face, or the jaw dropping would stretch them too.
2. **Field.** `face3d` turns the 468 displacements into a smooth field over
   the face, in two layers: one moves with the skull, one with the jaw. They
   are separate because the two lips lie a hair apart and move in opposite
   directions, and one field would average them into nothing at exactly the
   line the model most needs to move. The jaw's swing is taken out of the jaw
   layer first, because the models have hinges of their own (see below) and
   would otherwise open twice.
3. **Retarget.** A shark's mouth isn't where yours is, so each whole-head
   model has a thin-plate-spline warp fitted between features it really
   shares with a face: mouth corners, the lip line, eyes, top of head, chin.
   The face's motion is carried back through the inverse warp, so it arrives
   at the model's scale. A mouth corner that lifts a few millimetres lifts a
   gape three times the size of your mouth three times as far. Features a
   shark has no counterpart for (nose, brows, cheeks) are placed by the
   overall fit rather than matched. Matching them to invented points folded
   the warp and threw the teeth off the side of the head. Eyes, teeth,
   horns, tusks and ears ride the surface as rigid pieces rather than
   stretching with it. The painted animals' noses, snouts and tongues sit on
   the face itself, so they follow it with no warp at all.

Without a mesh (a bundle that has none, or the render harness) the models
fall back to the scores below, as before.

MediaPipe's Face Landmarker also returns the ARKit-style set of 52 blendshape
scores alongside the mesh. `Expression` (in `types.hpp`) carries the ones the
filters use from `detect()` through to the models. Where an articulation
needs a single number (the jaw's hinge, a lid closing over an eye, an ear
swinging) it comes from these:

| | driven by | what it does |
|---|---|---|
| jaw | `jawOpen` | the shark's and dinosaur's jaws hinge; the warps stretch the mouth |
| smile | `mouthSmile*` | rounds the squirrel's cheeks; lifts the elephant's ears and curls its trunk. The gape's own curl comes from the mesh (without one, from this score) |
| sad | `mouthFrown*` **and** `browInnerUp` | drops the corners, lays the ears back, and half-lids the eyes |
| blink | **eyelid landmarks**, not a blendshape | a lid slides down the eye of whichever model is worn |
| brows | `browInnerUp`, `browOuterUp*`, `browDown*` | the animals prick their ears up, or lay them back; the dinosaur stands its brow horns up or splays them |
| tongue | `tongueOut`, and the jaw | a tongue comes out — the shark's along the floor of its jaw, the animals' out of the muzzle |
| trunk | `jawOpen` | raises and curls the elephant's trunk |
| jaw sideways | `jawLeft` / `jawRight` | without a mesh, slides the shark's (or dinosaur's) lower jaw, teeth and tongue as one group; with one, the mesh's own sideways jaw motion does it |
| cheeks | `cheekPuff` | fills out the squirrel's cheek pouches |
| pucker | `mouthPucker` | curls the tip of the elephant's trunk |

Three things are worth knowing about how these are read:

- **Sides are the image's, not the subject's.** MediaPipe labels by anatomy,
  so which eye is on screen-left depends on whether the preview is mirrored.
  `detect()` settles that once from the eyes and puts every pair — landmarks
  and blendshapes alike — into screen order.
**A sad face is not one blendshape.** It is the corners of the mouth down
*and* the inner ends of the brows up, with the outer ends staying put — which
is what distinguishes it from surprise. `mouthFrown` alone misses half of it
and `browInnerUp` alone fires on plenty of things that are not sadness, so
`detect()` combines them once into `Expression::sad` and the models read that.

**Where an expression shows depends on the model.** The shark's mouth is most
of its face, so curling its mouth line is enough. The squirrel's is small and
under its muzzle and the elephant's is behind its trunk, so on those the same
expression has to go somewhere visible as well: the squirrel rounds its cheeks
and lays its ears back, the elephant lifts or drops its ears and curls its
trunk. Their mouth lines move only moderately, and for a structural reason —
on a head described by control points the mouth line also sets each
cross-section's upper and lower radii, so shifting it far does not curl the
mouth, it reshapes the head.

**Blinking slides a lid, it does not squash the eye.** An eyeball keeps its
shape when you blink; a lid slides over it. Squashing was the first attempt
and it failed twice: it scaled the bead's stand-off from the head along with
its height, so a shut eye sank into the surface and disappeared — and even
drawn, a slit is not what a closed eye looks like. The bead is left alone and
its *colour* is split instead: dark where the eyeball is still uncovered,
hide-coloured where a lid has come across it, with a dark lash line along the
lid's edge. Shut, that is a lid-shaped bulge with a line across it.

Eyes are also **solved onto the surface** rather than placed at a coordinate
that looks right from the front. Looking right from the front is not the same
as being on the head, and it was not: turn 40° and the far eye slid off the
side and hung in mid-air. Pick the depth and the height — the two things worth
controlling — and the sideways position falls out of the cross-section there.

- **Blink comes from the mesh, not the blendshape.** Eyelid aperture is just
  the gap between two landmarks over the eye's own width. That costs nothing,
  works on a bundle with no blendshape head at all, and does not wobble the
  way the predicted score does.
- **`tongueOut` is unreliable.** It is in MediaPipe's output, but the model
  seldom scores it above its noise floor, so nothing is built on it alone: the
  tongue also comes out when the jaw is simply open, which is true of a real
  mouth and always fires.

The first two filters *warp your actual face* — no cartoon mouth or eyes are pasted on
top; only the crying tears are drawn over the image.

**Two tables, one for each kind of filter.** A painted species is a `Coat` —
a marking function, a fur depth and how much shaggier the crown is. A
whole-head model is a `HeadShape` — the silhouette curves, the girth taper
and the countershading. Both replaced chains of ternaries that had stopped
being readable at three or four species, where adding another meant editing
several places at once.

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

The **Pi camera is mirrored by default**, like a phone's selfie camera: on this
build it is the IMX500 AI camera, facing the person holding it, and an
unmirrored preview moves the wrong way when you lean. `--mirror` picks which
cameras are flipped left-right — `picam` (the default), `webcam`, `all` or
`none` — and follows the camera switch button, so only the camera it names is
shown mirrored. Like `--camera-rotate` it applies to captures too, so a photo
is exactly the mirrored picture that was on screen. The flip is the last step,
after any facial filter, so it costs nothing on the GPU preview path and the
face tracking always sees the camera's own image.

```sh
build/open-lego-camera --mirror none   # show the Pi camera the way it sees
```

- Photos are saved as `IMG_YYYYMMDD_HHMMSS.jpg`. The gallery can still play
  back any existing `.mp4` videos in the output directory.
- The app opens on the **welcome screen**; tap **Start Camera** to begin or
  **Sleep** to switch the screen off (**double-tap** to wake). The **home** icon in
  the camera menu returns here.
- **Tap the screen** to wake the menu after it has faded.
- **Esc** or **Q** steps back one screen: camera → welcome, gallery → camera,
  and quits from the welcome screen. Any key wakes the screen from sleep.
  In the gallery, **←** / **→** step to older / newer items.
- `--windowed` is handy when developing on a desktop (the app then uses the
  desktop's SDL driver automatically).

## Sleep and power

**Sleep** on the welcome screen does more than draw black. A black frame alone
saves almost nothing: an LCD's backlight is most of its power draw and stays
lit behind black pixels, the display keeps scanning the frame out sixty times a
second, and the camera keeps streaming for a preview nobody is watching. So
sleep:

- **switches the backlight off** through the kernel's backlight class
  (`/sys/class/backlight/*`: `bl_power` to powered-down, brightness to 0),
  remembering the brightness to put back. That covers the HyperPixel's GPIO
  backlight and DSI panels such as the official 7″ and the Waveshare 5″;
- **switches the display pipeline off** (DPMS) through the DRM device SDL's
  `kmsdrm` driver already holds: scanout stops, a DSI/DPI panel's driver
  powers the panel down, and an HDMI monitor drops into standby. Needs libdrm
  at build time (`sudo apt install libdrm-dev`; CMake reports whether it
  found it) and no special permissions, since the app is already the
  display's owner;
- falls back to `vcgencmd display_power` only where neither of those exists
  (the legacy firmware display stack, where they don't apply);
- **closes the camera**, stopping the sensor, the ISP and the capture
  pipeline, and reopens the same camera at the same zoom on waking;
- **stops spinning**: the main loop blocks waiting for a touch instead of
  polling, waking once a second only to keep an eye on the battery (a
  low-battery shutdown still happens while asleep).

The touch controller is a separate device and stays on, so the
**double-tap** still wakes it. The app logs what it managed to switch off,
e.g. `display: switched off backlight 10-0045, display pipeline (DPMS)
(sleep)`.

**Backlight permissions.** The backlight files are root-only by default. The
app writes them directly if it can, and otherwise retries through `sudo -n`,
which on a stock Raspberry Pi OS (passwordless sudo for the default user)
just works. Without passwordless sudo, let the `video` group write them with
a udev rule, and make sure your user is in `video`:

```sh
sudo tee /etc/udev/rules.d/99-backlight.rules >/dev/null <<'RULE'
SUBSYSTEM=="backlight", ACTION=="add", RUN+="/bin/chgrp video /sys/class/backlight/%k/brightness /sys/class/backlight/%k/bl_power", RUN+="/bin/chmod g+w /sys/class/backlight/%k/brightness /sys/class/backlight/%k/bl_power"
RULE
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=backlight
```

In a desktop session (or `--windowed`) sleep only blanks the window: there the
display isn't the app's, and the backlight is the laptop's.

## Battery monitor (Waveshare UPS HAT)

With a Waveshare [UPS HAT (B)](https://www.waveshare.com/wiki/UPS_HAT_(B)) or
[UPS HAT (D)](https://www.waveshare.com/wiki/UPS_HAT_(D)) on the Pi's 40-pin
header, the app shows a **battery gauge in the top-right corner** of the
welcome, camera and gallery screens:

| Gauge | Meaning |
| --- | --- |
| green / amber / red fill + `NN%` | charge level (>50% / >20% / below) |
| blue fill with a **lightning bolt** | the pack is taking charge |
| pulsing red + `LOW BATTERY` | at or below 15%, off charge |

Nothing needs to be passed on the command line. The two models sit at different
I²C addresses, so at startup the app probes for each in turn and simply runs
without a gauge if neither answers:

| | UPS HAT (B) | UPS HAT (D) |
| --- | --- | --- |
| INA219 address | `0x42` | `0x43` |
| Pack | 2 × 18650 **in series** | 1 × 21700 |
| Empty → full | 7.0 V → 8.05 V (measured) | 3.0 V → 4.2 V (Waveshare's) |
| Shunt / profile | 0.1 Ω, 32 V / 2 A | 0.01 Ω, 16 V / 5 A |
| Power-path MCU | none | `0x2d` |

Two things have to be true on the Pi first:

```sh
# 1. I2C enabled -- add to /boot/firmware/config.txt (or use raspi-config), then reboot
dtparam=i2c_arm=on

# 2. your user allowed to use it (log out and back in afterwards)
sudo usermod -aG i2c "$USER"

# check the HAT is answering: 42 for a (B), or 43 (+ 2d) for a (D)
i2cdetect -y 1
```

Each board's register map, calibration and state-of-charge curve follow
Waveshare's own reference driver for that model, so the readings match what its
`INA219.py` demo prints. The per-model constants live in a single `kBoards`
table in [`src/battery.cpp`](src/battery.cpp) — adding another INA219-based HAT
means adding a row, not branching the driver.

### Calibrating the gauge to your pack

The percentage is **estimated from the pack's terminal voltage**, not counted in
coulombs, so it sags under a heavy load and recovers when the load drops; the
app smooths it so the number doesn't flicker.

That also means the endpoints matter, and **Waveshare's are optimistic**. Their
(B) formula assumes the pack swings the full 3.0–4.2 V per cell at the INA219's
terminals. It doesn't: the INA219 sits on the *load side* of the shunt, so every
reading is already down by the shunt drop plus the pack's own sag under load,
and the board stops delivering 5 V well before the cells are truly flat.
Measured on this camera, a full pack reads ~8.09 V and the Pi dies at ~6.96 V —
so the stock 6.0–8.4 V curve showed **87% on a full pack and 40% on a dead
one**. (Waveshare acknowledge the same skew in their FAQ, suggesting you fudge
the 6 down to 5.08.) The (B) default here is therefore the measured 7.0–8.05 V.

Your pack, cells and load will differ. To calibrate:

1. Charge fully, then run the camera on battery and note the voltage the app
   logs at startup — that's your **FULL**, minus a little headroom.
2. Run it flat and note the last voltage before it dies — that's your **EMPTY**.
3. Pass them back:

```bash
build/open-lego-camera --battery-range 7:8.05
```

The startup line shows the range in use, so you can confirm it took:

```
battery: Waveshare UPS HAT (B) @ /dev/i2c-1 0x42 (7.00-8.05 V)
```

The low-battery warning and the `--battery-shutdown` cut-off are both derived
from this range (the cut-off sits at 12.5% of it, which on the (D) works out to
exactly the 3.15 V Waveshare use), so recalibrating moves them with it instead
of stranding a threshold outside the new window.

Useful flags:

- `--battery-hat auto|b|d` — skip the probe and expect one specific model.
  Handy if something else on the bus answers at `0x42`/`0x43`.
- `--battery-bus N` — the HAT is on a bus other than `/dev/i2c-1`.
- `--no-battery` — skip the probe entirely.
- `--battery-range EMPTY:FULL` — calibrate the curve to your pack (above).
- `--battery-shutdown` — **opt-in**: when the pack stays below its cut-off
  (12.5% of the range: **7.13 V** on the (B), **3.15 V** on the (D)) for a
  minute while off charge, show `BATTERY EMPTY`, then halt. Without this flag
  the app only ever *reports* the level. On the (D) it first asks the power-path
  MCU to boot the Pi again by itself once the pack recovers (register `0x01` ←
  `0x55` at `0x2d`); the (B) has no such MCU, so it stays off until you press
  its button. The shutdown runs `sudo -n poweroff`, so it needs passwordless
  sudo (the default for the `pi` user) or root.

> Waveshare ship no low-voltage shutdown for the (B) at all, so its cut-off is
> this project's own, derived from the measured range above.

> The KiCad [CM4 carrier board](hardware/README.md) in this repo takes a
> different route — an on-board **MAX17048** fuel gauge — which this code does
> not read. The UPS HATs above are the off-the-shelf option for a normal
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
| camera in circling arrows | switch to the next camera (only shown when a second one is attached) |
| last-shot thumbnail (framed-landscape icon until the first capture) | open the gallery |
| ring with dot | take a photo (plays a shutter flash) |
| chevron (gallery) | back to the camera |
| ◀ / ▶ triangles (gallery) | older / newer item (the gallery opens on the newest, at the right end) |
| triangle-in-ring (gallery) | play the selected video |
| trash can (gallery) | delete the shown item (asks ✓ / ✗) |
| ✓ green / ✗ red | confirm / cancel a delete |

**Zoom** is not a button: **pinch with two fingers** on the preview to zoom
(digital, up to 4×). The current factor (`1.0x`–`4.0x`) appears briefly at the
top while you pinch.

In the gallery, the same pinch zooms the **photo** being viewed instead (up to
6×), growing out of the point between your fingers; while zoomed in, **drag
with one finger** to move around it. Stepping to another item resets the zoom.
Videos are not zoomed — what the gallery shows of one is just its first frame.

## Design notes

- **Modules** (`src/`): `camera` (dual backend + digital zoom), `gallery`
  (list/navigate/delete), `battery` (UPS HAT (B)/(D) INA219 over I²C), `icons`
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
